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
from tqdm.auto import tqdm, trange  # noqa
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, x):
        ctx.save_for_backward(x)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]

        factor = 1e-4 / (x.shape[0] * x.shape[1])

        maxx, ids = torch.max(x, -1, keepdim=True)

        gx = torch.zeros_like(x)
        gx.scatter_(-1, ids, maxx * factor)
        return (grad_output, gx)


from torch.utils.cpp_extension import load
import types
import math
T_MAX = 1024
HEAD_SIZE = 64
wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                                                  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])



@dataclass
class TrainConfig:
    # wandb params
    project: str = "D-RWKV"
    group: str = "Gym-MuJoCo"
    name: str = "DRWKV5-DCLAW"
    # model params
    embedding_dim: int = 128
    num_layers: int = 12
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "LRL"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 10000
    warmup_steps: int = 1000
    reward_scale: float = 1  # 1000 normalization for rewards/returns
    num_workers: int = 4
    # evaluation params

    eval_episodes: int = 1
    eval_every: int = 20000
    # general params
    checkpoints_path: Optional[str] = "/home/dong/LRL/dclaw/dclawModel"
    deterministic_torch: bool = False
    train_seed: int = 0
    eval_seed: int = 10
    device: str = "cuda"
    num_samples: int = 1000


class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):

        with torch.no_grad():

            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float,
                            memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.float
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float,
                             memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float,
                             memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float,
                             memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float,
                             memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float,
                             memory_format=torch.contiguous_format)  # .uniform_(-1, 1)
            wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            gw = torch.sum(gw, 0).view(H, C // H)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)


class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 当前layer id
        self.n_embd = TrainConfig.embedding_dim  # hidden_state 维度
        self.head_size = 64
        self.dim_att = TrainConfig.embedding_dim
        assert HEAD_SIZE == self.head_size  # change HEAD_SIZE to match args.head_size_a
        self.n_head = self.dim_att // self.head_size
        self.head_size_divisor = 8 #default

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (TrainConfig.num_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / TrainConfig.num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, TrainConfig.embedding_dim)
            for i in range(TrainConfig.embedding_dim):
                ddd[0, 0, i] = i / TrainConfig.embedding_dim
            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())
            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(TrainConfig.embedding_dim, self.dim_att, bias=False)
        self.key = nn.Linear(TrainConfig.embedding_dim, self.dim_att, bias=False)

        self.value = nn.Linear(TrainConfig.embedding_dim, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, TrainConfig.embedding_dim, bias=False)
        self.gate = nn.Linear(TrainConfig.embedding_dim, self.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att)


    @torch.jit.script_method
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @torch.jit.script_method
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):

        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # layer id

        # 平移
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / TrainConfig.num_layers))  # 1 to ~0

            x = torch.ones(1, 1, TrainConfig.embedding_dim)
            for i in range(TrainConfig.embedding_dim):
                x[0, 0, i] = i / TrainConfig.embedding_dim

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * TrainConfig.embedding_dim
        self.key = nn.Linear(TrainConfig.embedding_dim, hidden_sz, bias=False)  # 对应公式(17) 中的 W_k
        self.receptance = nn.Linear(TrainConfig.embedding_dim, TrainConfig.embedding_dim, bias=False)  # 对应公式(16) 中的 W_r
        self.value = nn.Linear(hidden_sz, TrainConfig.embedding_dim, bias=False)  # 对应公式(18) 中的 W_v

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv  # 公式（18）中
        return rkv

# general utils
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

# some utils functionalities specific for Decision Transformer
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
    info_path = "/home/dong/LRL/dclaw/dataset_SAC_DClaw/"+env_name+"_info.pkl"
    traj_path = "/home/dong/LRL/dclaw/dataset_SAC_DClaw/" + env_name + "_traj.pkl"
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
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]

        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["states"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
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

# Decision Transformer implementation

class Block(nn.Module):
    """一个RWKV块"""

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 当前layer的id
        self.ln1 = nn.LayerNorm(TrainConfig.embedding_dim)
        self.ln2 = nn.LayerNorm(TrainConfig.embedding_dim)
        self.Time_mix = RWKV_TimeMix(layer_id)
        self.Channel_mix = RWKV_ChannelMix(layer_id)

    def forward(self, x):

        x = x + self.Time_mix(self.ln1(x))
        x = x + self.Channel_mix(self.ln2(x))
        return x


class DecisionRWKV(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 1024,
        episode_len: int = 1000,
        embedding_dim: int = TrainConfig.embedding_dim,
        num_layers: int = TrainConfig.num_layers,
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
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList([Block(action_dim) for _ in range(TrainConfig.num_layers)])

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
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        # print(states)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        for block in self.blocks:
            out = block(out)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action

        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionRWKV,
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
    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        # print(actions[:, : step + 1][:, -model.seq_len :])
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        # print(predicted_actions)
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)

        # print(reward)
        score += reward
        # env.render()
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
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
        self.buffer = defaultdict(list)  # 使用字典来存储不同任务的经验

    def add(self, task_id, experience):
        task_buffer = self.buffer[task_id]
        if len(task_buffer) >= self.buffer_size:
            task_buffer.pop(0)
        task_buffer.append(experience)

    def sample(self, batch_size):
        samples = []
        per_task_batch_size = max(1, batch_size // len(self.buffer))  # 确保每个任务至少有一个样本
        for task_id, task_buffer in self.buffer.items():
            task_samples = random.sample(task_buffer, min(len(task_buffer), per_task_batch_size))
            samples.extend(task_samples)
        return samples[:batch_size]  # 如果样本过多，只取需要的数量


@pyrallis.wrap()
def train(config: TrainConfig, model, envName, replay_buffer, task_id):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    plot_X = []
    plot_Y = []
    # data & dataloader setup
    dataset = SequenceDataset(
        envName, seq_len=config.seq_len, reward_scale=config.reward_scale
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)

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
    # save config to the checkpoint

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        for i in range(states.shape[0]):
            replay_buffer.add(task_id, (states[i], actions[i], returns[i], time_steps[i], mask[i]))
        # 从回放缓冲区中采样经验
        replay_samples = replay_buffer.sample(config.batch_size // 2)  # 假设一半的数据来自回放缓冲区
        replay_states, replay_actions, replay_returns, replay_time_steps, replay_mask = map(
            lambda x: torch.stack(x).to(config.device), zip(*replay_samples)
        )

        # 将当前任务的数据和回放缓冲区的数据合并
        states = torch.cat((states, replay_states), dim=0)
        actions = torch.cat((actions, replay_actions), dim=0)
        returns = torch.cat((returns, replay_returns), dim=0)
        time_steps = torch.cat((time_steps, replay_time_steps), dim=0)
        mask = torch.cat((mask, replay_mask), dim=0)

        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()
        # print("loss:",loss)
        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        # validation in the env for the actual online performance
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
    # Initialize the model once and train on multiple environments
    model = DecisionRWKV(
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
    # Train on each environment in the list
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
