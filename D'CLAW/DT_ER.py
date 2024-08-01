import itertools
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
import robel
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  
import pickle
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class TrainConfig:
    project: str = "D-TRANSFORMER"
    group: str = "DT-D4RL"
    name: str = "DT-DCLAW"
    embedding_dim: int = 128
    num_layers: int = 12
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    env_name: str = "LRL"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 10000
    warmup_steps: int = 1000
    reward_scale: float = 1  
    num_workers: int = 4

    eval_episodes: int = 1
    eval_every: int = 20000
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 42
    eval_seed: int = 10
    device: str = "cuda"
    num_samples: int = 1000


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

def load_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    info_path = "dataset_SAC_DClaw/"+env_name+"_info.pkl"
    traj_path = "dataset_SAC_DClaw/" + env_name + "_traj.pkl"
    with open(info_path, "rb") as f:
            info = pickle.load(f)
    with open(traj_path, "rb") as f:
            traj = pickle.load(f)
    return traj, info

class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 20, reward_scale: float = 1.0):
        self.dataset, info = load_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len
        self.num_samples = TrainConfig.num_samples

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]

        states = traj["states"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]

        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 20,
        episode_len: int = 300,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  
        actions: torch.Tensor,  
        returns_to_go: torch.Tensor,  
        time_steps: torch.Tensor,  
        padding_mask: Optional[torch.Tensor] = None,  
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out = self.action_head(out[:, 1::3]) * self.max_action

        return out


@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )

    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)
    score = 0
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):

        predicted_actions = model(  
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )

        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)

        score += reward
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return score

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = defaultdict(list)  

    def add(self, task_id, experience):
        task_buffer = self.buffer[task_id]
        if len(task_buffer) >= self.buffer_size:
            task_buffer.pop(0)
        task_buffer.append(experience)

    def sample(self, batch_size):
        samples = []
        per_task_batch_size = max(1, batch_size // len(self.buffer))  
        for task_id, task_buffer in self.buffer.items():
            task_samples = random.sample(task_buffer, min(len(task_buffer), per_task_batch_size))
            samples.extend(task_samples)
        return samples[:batch_size]  


@pyrallis.wrap()
def train(config: TrainConfig, model, envName, replay_buffer, task_id):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    plot_X = []
    plot_Y = []
    dataset = SequenceDataset(
        envName, seq_len=config.seq_len, reward_scale=config.reward_scale
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )


    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        for i in range(states.shape[0]):
            replay_buffer.add(task_id, (states[i], actions[i], returns[i], time_steps[i], mask[i]))

        replay_samples = replay_buffer.sample(config.batch_size // 2)  
        replay_states, replay_actions, replay_returns, replay_time_steps, replay_mask = map(
            lambda x: torch.stack(x).to(config.device), zip(*replay_samples)
        )

        states = torch.cat((states, replay_states), dim=0)
        actions = torch.cat((actions, replay_actions), dim=0)
        returns = torch.cat((returns, replay_returns), dim=0)
        time_steps = torch.cat((time_steps, replay_time_steps), dim=0)
        mask = torch.cat((mask, replay_mask), dim=0)

        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        loss = (loss * mask.unsqueeze(-1)).mean()
        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        if step % config.eval_every == 0 or step == config.update_steps-1:
            model.eval()

            env_list = ["DClawTurnFixedT0-v0", "DClawTurnFixedT1-v0", "DClawTurnFixedT2-v0", "DClawTurnFixedT3-v0",
                        "DClawTurnFixedT4-v0", "DClawTurnFixedT5-v0", "DClawTurnFixedT6-v0", "DClawTurnFixedT7-v0",
                        "DClawTurnFixedT8-v0", "DClawTurnFixedT9-v0"]
            scores = []
            score_sum = 0
            for env_name in env_list:
                eval_dataset = SequenceDataset(
                    env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
                )
                eval_env = wrap_env(
                    env=gym.make(env_name),
                    state_mean=eval_dataset.state_mean,
                    state_std=eval_dataset.state_std,
                    reward_scale=config.reward_scale,
                )
                eval_env.seed(config.eval_seed)
                score = eval_rollout(
                    model=model,
                    env=eval_env,
                    target_return=12 * config.reward_scale,
                    device=config.device,
                )
                score_sum = score_sum+score
                scores.append(score)
            plot_X.append(step)
            plot_Y.append(scores)
            print()
            print("avg_score:", score_sum/10)
            print(scores)

            model.train()


    return plot_X, plot_Y


@pyrallis.wrap()
def train_all(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    config.state_dim = 21
    config.action_dim = 9
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)
    env_list = ["DClawTurnFixedT0-v0", "DClawTurnFixedT1-v0", "DClawTurnFixedT2-v0", "DClawTurnFixedT3-v0",
                "DClawTurnFixedT4-v0", "DClawTurnFixedT5-v0", "DClawTurnFixedT6-v0", "DClawTurnFixedT7-v0",
                "DClawTurnFixedT8-v0", "DClawTurnFixedT9-v0"]
    replay_buffer = ReplayBuffer(buffer_size=10000)
    plot_X = []
    plot_Y = []

    for idx, envName in enumerate(env_list):
        X, Y = train(model=model, envName=envName, replay_buffer=replay_buffer, task_id=idx)
        plot_X.append(X)
        plot_Y.append(Y)
    np.save(f"{TrainConfig.name}-{TrainConfig.embedding_dim}-{TrainConfig.num_layers}-{TrainConfig.train_seed}"+'-plot_X.npy', np.array(plot_X))
    np.save(f"{TrainConfig.name}-{TrainConfig.embedding_dim}-{TrainConfig.num_layers}-{TrainConfig.train_seed}"+'-plot_Y.npy', np.array(plot_Y))
    if config.checkpoints_path is not None:
        checkpoints_path = f"{TrainConfig.name}-{TrainConfig.embedding_dim}-{TrainConfig.num_layers}-{TrainConfig.train_seed}"
        torch.save(model.state_dict(), os.path.join(config.checkpoints_path,checkpoints_path+"-checkpoint.pt"))

        print("saving....checkpoint in" + os.path.join(config.checkpoints_path, checkpoints_path+"-checkpoint.pt"))

if __name__ == "__main__":
    TrainConfig.train_seed = 0
    train_all()
    TrainConfig.train_seed = 10
    train_all()
    TrainConfig.train_seed = 42
    train_all()
