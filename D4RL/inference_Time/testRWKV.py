
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa
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
import subprocess
import re

def get_gpu_memory_usage():
    # 执行nvidia-smi命令获取GPU状态
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # 解码输出
    output = result.stdout.decode('utf-8')

    # 使用正则表达式解析显存使用情况
    gpu_memory_info = re.findall(r'(\d+),\s*(\d+)', output)

    # 将字符串转换为整数，并以列表形式返回
    gpu_memory_usage = [{'total_memory': int(total), 'used_memory': int(used)} for total, used in gpu_memory_info]

    return gpu_memory_usage

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
@dataclass
class TrainConfig:
    # wandb params
    project: str = "D-RWKV"
    group: str = "Gym-MuJoCo"
    name: str = "halfcheetah-medium-v2"
    # model params
    embedding_dim: int = 128
    num_layers: int = 12
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 100
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "halfcheetah-medium-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 500
    warmup_steps: int = 100
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 1
    eval_every: int = 100
    # general params
    checkpoints_path: Optional[str] = "/home/yujiandong/LRL/d4rl/model/"
    deterministic_torch: bool = False
    train_seed: int = 0
    eval_seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.p
from torch.utils.cpp_extension import load
import types
import math
T_MAX = 204800000
wkv_cuda = load(name="wkv", sources=["cuda/wkv4_op.cpp", "cuda/wkv4_cuda.cu"],
                verbose=True, extra_cuda_cflags=
                ['-res-usage', '--use_fast_math', '-O3','--maxrregcount=60', '-Xptxas=-O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C

        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 当前layer id
        self.ctx_len = TrainConfig.seq_len  # 最长文本长度
        self.n_embd = TrainConfig.embedding_dim  # hidden_state 维度

        with torch.no_grad():  # fancy init

            ratio_0_to_1 = (layer_id / (TrainConfig.num_layers - 1))  # 0 to 1   w的  l / (L - 1)

            ratio_1_to_almost0 = (1.0 - (layer_id / TrainConfig.num_layers))  # 1 to ~0   u(mu)的  1-（l/L）

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)  # 维度的位置编码 [hidden_state_size]
            for h in range(self.n_embd):  # 按隐藏维度循环每一个位置

                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first 对应 论文中的bonus

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix 对应公式中的(11-13)
            x = torch.ones(1, 1, TrainConfig.embedding_dim)
            for i in range(TrainConfig.embedding_dim):

                x[0, 0, i] = i / TrainConfig.embedding_dim
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 对应 U(mu)_ki
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 对应 U(mu)_Vi
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))  # 对应 U(mu)_ri

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # 定义 Wr Wk Wv
        self.key = nn.Linear(TrainConfig.embedding_dim, self.n_embd, bias=False)
        self.value = nn.Linear(TrainConfig.embedding_dim, self.n_embd, bias=False)
        self.receptance = nn.Linear(TrainConfig.embedding_dim, self.n_embd, bias=False)

        # 定义 Wo
        self.output = nn.Linear(self.n_embd, TrainConfig.embedding_dim, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):
        """C++ 调用"""
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)  # X_t-1
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 公式 (12) 中的 括号部分
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 公式 (13) 中的 括号部分
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 公式 (11) 中的 括号部分

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)  # 公式 (12) 中的K_t
        v = self.value(xv)  # 公式 (13) 中的V_t
        r = self.receptance(xr)  # 公式 (11) 中的R_t
        sr = torch.sigmoid(r)  # 公式 (15) 中的sigmoid_Rt

        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)  <=>   batch_size sentence_len hidden_size
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)  # 对应公式 (15)
        return rwkv


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


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()

    traj, traj_len = [], []
    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }

    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 20, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
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


import torch
import torch.nn as nn
import time
from typing import Optional

# 假设DecisionRWKV类已经定义好了，这里省略了类的定义

def test_decision_transformer_time(
    state_dim: int,
    action_dim: int,
    seq_len: int,
    episode_len: int,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    device:str,
):
    # 创建模型实例DecisionRWKV
    model = DecisionRWKV(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        episode_len=episode_len,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    model.eval()  # 设置为评估模式

    # 创建随机输入数据
    states = torch.randn(batch_size, seq_len, state_dim, dtype=torch.float, device=device)
    actions = torch.randn(batch_size, seq_len, action_dim, dtype=torch.float, device=device)
    returns_to_go = torch.randn(batch_size, seq_len, dtype=torch.float, device=device)
    time_steps = torch.randint(0, episode_len, (batch_size, seq_len), device=device)
    with torch.no_grad():  # 确保不计算梯度
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        _ = model(states, actions, returns_to_go, time_steps)
        end_event.record()

        torch.cuda.synchronize(device)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        gpu_memory = get_gpu_memory_usage()
        for idx, gpu in enumerate(gpu_memory):
            print(f"GPU {idx}: Total Memory: {gpu['total_memory']}B, Used Memory: {gpu['used_memory']} MiB")
            memory_used = gpu['used_memory']

    print(elapsed_time_ms,memory_used)

    return elapsed_time_ms, memory_used

import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size = TrainConfig.batch_size
    state_dim = 32
    action_dim = 4
    episode_len = TrainConfig.episode_len
    embedding_dim = TrainConfig.embedding_dim
    num_layers = TrainConfig.num_layers
    num_heads = TrainConfig.num_heads
    device = TrainConfig.device
    # 初始化记录时间和序列长度的列表
    average_times = []
    sequence_lengths = []
    memorycost = []
    seq_len= 7000
    # 对每个序列长度seq_len运行测试


    time_taken,memory_cost = test_decision_transformer_time(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        episode_len=episode_len,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        batch_size=batch_size,
        device = device
    )
