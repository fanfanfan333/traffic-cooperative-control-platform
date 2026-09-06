"""
交通信号控制 DQN 智能体
━━━━━━━━━━━━━━━━━━━━━━━━
状态空间  : 车道车辆密度 + 队列长度 + 速度分布 + 当前相位 + 相位时长
动作空间  : 下一相位选择（离散）
奖励函数  : 负累积等待时间 + 吞吐量 + 紧急惩罚
经验回放  : Prioritized Experience Replay (PER) + n-step回报(默认3步)
网络结构  : Dueling Double DQN
学习率    : 默认3e-3,线性warmup + 余弦退火(高学习率快速学习,后期精细收敛)
稳定器    : Huber损失 + 梯度裁剪 + 目标网络Polyak软更新 + loss熔断保护
"""

import math
import numpy as np
import random
from collections import deque, namedtuple
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. 交通状态空间构建
# ─────────────────────────────────────────────

class TrafficStateBuilder:
    """
    从 DeepSORT 追踪结果构建 DQN 状态向量

    路口拓扑（可配置）：
      4 条进道（North / South / East / West）
      每条进道 2 车道
      => 8 个车道 × (密度 + 队列 + 平均速度) = 24 个基础特征

    附加特征：
      当前信号相位（one-hot, 4 相位） = 4
      当前相位已持续帧数（归一化）     = 1
      ──────────────────────────────────
      总状态维度：29
    """

    PHASES = {
        0: "NS_Green",    # 南北直行
        1: "NS_Left",     # 南北左转
        2: "EW_Green",    # 东西直行
        3: "EW_Left",     # 东西左转
    }

    LANE_REGIONS = {
        # 车道 ID → ROI 区域 [x1, y1, x2, y2]（归一化 0-1，相对帧尺寸）
        "N_L1": [0.40, 0.00, 0.50, 0.45],
        "N_L2": [0.50, 0.00, 0.60, 0.45],
        "S_L1": [0.40, 0.55, 0.50, 1.00],
        "S_L2": [0.50, 0.55, 0.60, 1.00],
        "E_L1": [0.55, 0.40, 1.00, 0.50],
        "E_L2": [0.55, 0.50, 1.00, 0.60],
        "W_L1": [0.00, 0.40, 0.45, 0.50],
        "W_L2": [0.00, 0.50, 0.45, 0.60],
    }
    LANE_NAMES = list(LANE_REGIONS.keys())

    # 停止线 y 坐标（归一化），用于判断是否"在队列中"
    STOP_LINE_Y = {"N": 0.45, "S": 0.55, "E": 0.50, "W": 0.50}
    SPEED_THRESHOLD = 2.0   # px/frame，低于此视为静止

    def __init__(self, frame_w: int = 1280, frame_h: int = 720,
                 max_vehicles_per_lane: int = 20):
        self.W = frame_w
        self.H = frame_h
        self.max_v = max_vehicles_per_lane
        self.state_dim = len(self.LANE_NAMES) * 3 + len(self.PHASES) + 1

    def build(
        self,
        tracks:       list[dict],
        current_phase: int,
        phase_duration: int,
        max_phase_duration: int = 120,
    ) -> np.ndarray:
        """
        Args:
            tracks          : DeepSORT 输出列表
            current_phase   : 当前信号相位索引 (0-3)
            phase_duration  : 当前相位已持续帧数
            max_phase_duration: 最大相位时长（用于归一化）

        Returns:
            state: np.ndarray (state_dim,)，所有特征归一化到 [0, 1]
        """
        # ── 初始化各车道容器 ──
        lane_vehicles: dict[str, list] = {k: [] for k in self.LANE_NAMES}

        # ── 将追踪目标分配到对应车道 ──
        for trk in tracks:
            bbox  = trk["bbox"]           # [x1, y1, x2, y2]
            vx, vy = trk["velocity"]

            cx = ((bbox[0] + bbox[2]) / 2) / self.W  # 归一化中心 x
            cy = ((bbox[1] + bbox[3]) / 2) / self.H  # 归一化中心 y

            lane = self._assign_lane(cx, cy)
            if lane:
                speed = np.sqrt(vx**2 + vy**2)
                lane_vehicles[lane].append({
                    "cx": cx, "cy": cy, "speed": speed,
                    "class_id": trk["class_id"],
                })

        # ── 逐车道提取特征 ──
        lane_features = []
        for lane_name in self.LANE_NAMES:
            vehicles = lane_vehicles[lane_name]
            direction = lane_name[0]   # N / S / E / W

            density = len(vehicles) / self.max_v           # 密度（归一化）

            # 队列长度：统计速度 < 阈值的车辆数
            queued = sum(1 for v in vehicles if v["speed"] < self.SPEED_THRESHOLD)
            queue_len = queued / self.max_v

            # 平均速度（归一化，最大速度假设 30 px/frame）
            if vehicles:
                avg_speed = np.mean([v["speed"] for v in vehicles]) / 30.0
            else:
                avg_speed = 0.0

            lane_features.extend([
                np.clip(density,   0, 1),
                np.clip(queue_len, 0, 1),
                np.clip(avg_speed, 0, 1),
            ])

        # ── 相位 one-hot ──
        phase_onehot = np.zeros(len(self.PHASES), dtype=np.float32)
        phase_onehot[current_phase] = 1.0

        # ── 相位持续时间（归一化）──
        phase_norm = np.clip(phase_duration / max_phase_duration, 0, 1)

        state = np.array(lane_features, dtype=np.float32)
        state = np.concatenate([state, phase_onehot, [phase_norm]])
        return state

    def _assign_lane(self, cx: float, cy: float) -> Optional[str]:
        """根据归一化坐标 (cx, cy) 返回车道名称，不在任何车道则返回 None"""
        for lane_name, (x1, y1, x2, y2) in self.LANE_REGIONS.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return lane_name
        return None


# ─────────────────────────────────────────────
# 2. 奖励函数设计
# ─────────────────────────────────────────────

class RewardCalculator:
    """
    多目标奖励函数
    r = -α·ΔWait - β·ΔQueue + γ·Throughput - δ·EmergencyPenalty
    """

    def __init__(
        self,
        alpha: float = 1.0,    # 等待时间权重
        beta:  float = 0.5,    # 队列长度权重
        gamma: float = 2.0,    # 吞吐量权重
        delta: float = 5.0,    # 紧急情况惩罚权重
        max_wait_reward: float = 10.0,   # 奖励值裁剪上限
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.clip  = max_wait_reward

        # 内部状态
        self._prev_total_wait  = 0.0
        self._prev_total_queue = 0.0
        self._passed_count     = 0      # 本步通过停止线车辆数（吞吐量）

    def reset(self):
        self._prev_total_wait  = 0.0
        self._prev_total_queue = 0.0
        self._passed_count     = 0

    def compute(
        self,
        state:      np.ndarray,
        next_state: np.ndarray,
        throughput: int   = 0,     # 本时间步通过路口车辆数
        emergency:  bool  = False, # 是否有紧急车辆被阻塞
    ) -> float:
        """
        计算单步奖励

        使用状态向量中的密度/队列分量估计等待代价
        （真实部署中可从仿真环境获取精确数值）
        """
        # 从状态向量解析特征（每3个为一组: density, queue, speed）
        n_lanes = 8
        densities_now  = state     [0:n_lanes*3:3]
        queues_now     = state     [1:n_lanes*3:3]
        densities_next = next_state[0:n_lanes*3:3]
        queues_next    = next_state[1:n_lanes*3:3]

        total_wait_now  = float(np.sum(densities_now))
        total_wait_next = float(np.sum(densities_next))
        total_queue_now  = float(np.sum(queues_now))
        total_queue_next = float(np.sum(queues_next))

        # 等待时间变化（负值 = 等待减少 = 好）
        delta_wait  = total_wait_next  - total_wait_now
        delta_queue = total_queue_next - total_queue_now

        reward = (
            - self.alpha * delta_wait
            - self.beta  * delta_queue
            + self.gamma * throughput
            - self.delta * int(emergency)
        )

        return float(np.clip(reward, -self.clip, self.clip))


# ─────────────────────────────────────────────
# 3. 优先经验回放 (PER)
# ─────────────────────────────────────────────

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class PrioritizedReplayBuffer:
    """
    基于 SumTree 的优先级经验回放
    采样概率 ∝ |TD误差|^α
    重要性采样权重 w_i = (N·P(i))^{-β}
    """

    def __init__(
        self,
        capacity: int   = 50_000,
        alpha:    float = 0.6,    # 优先级指数
        beta:     float = 0.4,    # 重要性采样指数（随训练线性增大到 1.0）
        beta_increment: float = 0.001,
        epsilon:  float = 1e-6,   # 防零
    ):
        self.capacity = capacity
        self.alpha    = alpha
        self.beta     = beta
        self.beta_inc = beta_increment
        self.epsilon  = epsilon

        self._tree    = np.zeros(2 * capacity - 1, dtype=np.float64)  # SumTree
        self._data    = np.empty(capacity, dtype=object)
        self._ptr     = 0
        self._size    = 0
        self._max_pri = 1.0

    # ── SumTree 操作 ──
    def _tree_update(self, idx: int, priority: float):
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] += change

    def _tree_sample(self, value: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self._tree):
                break
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self._data[data_idx]

    @property
    def total_priority(self) -> float:
        return self._tree[0]

    def push(self, transition: Transition):
        """存入新经验，赋予当前最高优先级。
        注意: SumTree叶子节点里存的优先级在 update_priorities 里已经做过
        **alpha 幂运算, _max_pri 记录的就是树内最大值, 这里直接取用;
        旧版这里又乘了一次 **alpha(相当于alpha的平方),会让新样本的
        优先级系统性低于真正的最大值、几乎采样不到,属于bug,已修正。"""
        priority = self._max_pri
        tree_idx = self._ptr + self.capacity - 1
        self._data[self._ptr] = transition
        self._tree_update(tree_idx, priority)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Returns:
            transitions: list[Transition]
            indices    : list[int] (SumTree 索引，用于更新优先级)
            weights    : np.ndarray (重要性采样权重)
        """
        segment = self.total_priority / batch_size
        transitions, indices, weights = [], [], []

        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            val = random.uniform(a, b)
            idx, trans = self._tree_sample(val)
            # 防御: 浮点边界情况可能恰好命中未写入的0优先级叶子,
            # 此时退化为从已存经验里随机取一条,保证batch始终有效
            if trans is None:
                data_idx = random.randrange(self._size)
                trans = self._data[data_idx]
                idx = data_idx + self.capacity - 1
            indices.append(idx)
            transitions.append(trans)

            prob = self._tree[idx] / self.total_priority
            prob = max(prob, self.epsilon)
            w = (prob * self._size) ** (-self.beta)
            weights.append(w)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()   # 归一化
        return transitions, indices, weights

    def update_priorities(self, indices: list[int], td_errors: np.ndarray):
        """训练后用新 TD 误差更新优先级"""
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.epsilon) ** self.alpha
            self._max_pri = max(self._max_pri, p)
            self._tree_update(idx, p)

    def __len__(self) -> int:
        return self._size


# ─────────────────────────────────────────────
# 4. Dueling Double DQN 网络（PyTorch 完整反传）
# ─────────────────────────────────────────────

class DuelingDQNNetwork(nn.Module):
    """V(s) + A(s,a) - mean(A)，共享层与两路输出都会被梯度更新。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.value = nn.Linear(in_dim, 1)
        self.advantage = nn.Linear(in_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.shared(state)
        V = self.value(feat)
        A = self.advantage(feat)
        return V + (A - A.mean(dim=-1, keepdim=True))


# ─────────────────────────────────────────────
# 4.5 学习率调度: 线性 warmup + 余弦退火
# ─────────────────────────────────────────────

class LinearWarmupCosineLR:
    """
    让"高学习率"既快又稳的调度器:
      1) 前 warmup_steps 步从接近0线性升到 base_lr
         —— 避免训练刚冷启动、Q值估计还很离谱时就被大学习率猛踢一脚
            (高学习率训练发散最常发生在刚开始的阶段);
      2) 之后按余弦曲线从 base_lr 平滑退火到 base_lr * min_lr_scale
         —— 前期大学习率快速逼近好策略(学习速度快),
            后期小学习率精细收敛(抖动小、不破坏已学到的策略)。
    """

    def __init__(self, optimizer, base_lr: float,
                 warmup_steps: int = 500, total_steps: int = 50_000,
                 min_lr_scale: float = 0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(0, int(total_steps))
        self.min_lr = base_lr * min_lr_scale
        self._step = 0

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.total_steps <= 0:
            return self.base_lr          # 不退火,只做warmup
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self) -> float:
        """把当前进度对应的学习率写进优化器,返回该学习率"""
        lr = self.lr_at(self._step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> float:
        """每完成一次参数更新调用一次,推进调度进度"""
        self._step += 1
        return self.apply()

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict):
        self._step = int(state.get("step", 0))


# ─────────────────────────────────────────────
# 5. DQN 智能体
# ─────────────────────────────────────────────

class TrafficDQNAgent:
    """
    交通信号 Dueling Double DQN 智能体
    动作空间：选择下一信号相位（4 个相位）
    """

    ACTION_SPACE = list(TrafficStateBuilder.PHASES.keys())   # [0, 1, 2, 3]

    def __init__(
        self,
        state_dim:      int   = 29,
        action_dim:     int   = 4,
        lr:             float = 3e-3,    # 学习率: 从1e-3提高到3e-3(配合warmup+余弦退火,收敛更快且稳定)
        gamma:          float = 0.95,    # 折扣因子
        epsilon:        float = 1.0,     # 初始探索率
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 128,     # 批量从64提高到128,梯度噪声更小,支撑更高学习率
        target_update_freq: int = 200,   # 目标网络硬同步频率(步),仅在 target_tau<=0 时使用
        target_tau:     float = 0.005,   # 目标网络Polyak软更新系数; <=0 时退回硬同步
        replay_capacity: int  = 50_000,
        min_replay_size: int  = 1_000,   # 开始训练的最小经验量
        hidden_dims:    tuple = (128, 64),
        n_step:         int   = 3,       # n-step回报视野; 设1退化为普通1-step
        lr_warmup_steps: int  = 500,     # 学习率线性warmup步数(0=不做warmup)
        lr_total_steps: int  = 50_000,   # 余弦退火总步数(0=不退火,保持base_lr)
        lr_min_scale:   float = 0.1,     # 退火终点学习率 = lr * 此系数
        use_huber:      bool  = True,    # Huber损失,抗离群TD误差,是高学习率的关键稳定器
    ):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.lr          = lr
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update_freq = target_update_freq
        self.target_tau  = target_tau
        self.min_replay  = min_replay_size
        self.hidden_dims = tuple(hidden_dims)
        self.n_step      = max(1, int(n_step))
        self.use_huber   = use_huber

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.lr_scheduler = None
        self.current_lr = lr
        if lr_warmup_steps > 0 or lr_total_steps > 0:
            self.lr_scheduler = LinearWarmupCosineLR(
                self.optimizer, base_lr=lr,
                warmup_steps=lr_warmup_steps, total_steps=lr_total_steps,
                min_lr_scale=lr_min_scale,
            )
            self.current_lr = self.lr_scheduler.apply()

        self.replay = PrioritizedReplayBuffer(capacity=replay_capacity)

        # n-step滑动窗口: 暂存最近几条原始经验,凑满n条聚合成一条n-step经验入池
        self._pending: deque = deque()

        self._step_count = 0
        self.loss_history: deque = deque(maxlen=1000)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.choice(list(range(self.action_dim)))

        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(x).squeeze(0).cpu().numpy()
        return int(np.argmax(q_values))

    # ─────────────── 经验存储(n-step) ───────────────

    def store(self, state, action, reward, next_state, done):
        """
        存入一条原始经验。内部维护 n-step 滑动窗口:
        每凑满 n 条就把队首聚合成一条 n-step 经验推入回放池
        (回报为 n 步折扣累加,bootstrap 到第 n 步的状态),
        让奖励信号更快地传播到更早的决策,显著加快信用分配。
        done=True 时立即把剩余尾巴全部冲进回放池并截断。
        """
        self._pending.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))
        if done:
            self.flush_replay()
        elif len(self._pending) >= self.n_step:
            self._push_streaming()

    def _push_streaming(self):
        """窗口滑动一步: 聚合队首n条为一条n-step经验入池,只弹出队首1条"""
        n = self.n_step
        discounted_return = sum((self.gamma ** k) * self._pending[k][2] for k in range(n))
        next_state = self._pending[n - 1][3]
        self.replay.push(Transition(
            self._pending[0][0], self._pending[0][1], discounted_return, next_state, False,
        ))
        self._pending.popleft()

    def flush_replay(self):
        """
        episode 结束时调用: 把滑动窗口里剩余不足n条的尾巴也聚合成
        截断的 n-step 经验推入回放池(折扣累加到done为止),保证最近的
        经验不丢失。done之后如果还有残余经验则丢弃(终态之后无意义)。
        """
        while self._pending:
            horizon = min(self.n_step, len(self._pending))
            discounted_return, next_state, done, effective = 0.0, None, False, horizon
            for k in range(horizon):
                discounted_return += (self.gamma ** k) * self._pending[k][2]
                next_state = self._pending[k][3]
                if self._pending[k][4]:
                    done, effective = True, k + 1
                    break
            self.replay.push(Transition(
                self._pending[0][0], self._pending[0][1], discounted_return, next_state, done,
            ))
            for _ in range(effective):
                self._pending.popleft()
            if done:
                self._pending.clear()

    # ─────────────── 训练步骤 ───────────────

    def train_step(self) -> Optional[float]:
        """
        执行一次 Dueling Double DQN 训练步骤

        Returns:
            loss (float) 或 None（经验不足时）
        """
        if len(self.replay) < self.min_replay:
            return None

        transitions, tree_indices, is_weights = self.replay.sample(self.batch_size)

        states      = np.array([t.state      for t in transitions], dtype=np.float32)
        actions     = np.array([t.action     for t in transitions], dtype=np.int32)
        rewards     = np.array([t.reward     for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones       = np.array([t.done       for t in transitions], dtype=np.float32)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        is_w_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        td_errors = target_q - current_q
        # Huber(平滑L1)损失: 小误差区间等价于平方、大误差区间退化为线性,
        # 对离群TD误差的梯度有上界。学习率提高到3e-3后,这是防止个别
        # 极端样本的大梯度把训练带飞的关键稳定器(旧版纯MSE配合高学习率
        # 极易在冷启动阶段发散)。
        if self.use_huber:
            per_sample_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        else:
            per_sample_loss = td_errors.pow(2)
        loss = (is_w_t * per_sample_loss).mean()

        loss_val = float(loss.item())
        if not np.isfinite(loss_val):
            # 熔断保护: loss 出现 NaN/Inf 说明这一批更新数值炸了
            # (常见诱因: 优化器状态和新超参数不匹配、reward量级突变、
            # 极端outlier样本等)。这种情况下绝不能让 backward()+step()
            # 真的执行,否则一次异常更新就会把网络权重永久污染成NaN,
            # 后续所有决策都会崩坏(实测中出现过loss=nan之后,
            # 车均延误从100多秒直接飙到500~1000+秒、相位分布严重偏科的情况)。
            # 这里直接跳过这次参数更新,只打印警告,不让坏梯度进网络。
            print(f"[DQN] 警告: 检测到 loss={loss_val},跳过本次参数更新以避免网络被污染")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.current_lr = self.lr_scheduler.step()

        self._step_count += 1
        if self.target_tau > 0:
            # Polyak软更新: θ_target ← (1-τ)·θ_target + τ·θ_online。
            # 相比每200步一次性硬拷贝,软更新让训练目标平滑漂移,
            # 学习率提高后目标Q值的抖动明显更小,收敛更稳。
            self._soft_target_update()
        elif self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.replay.update_priorities(tree_indices, np.abs(td_errors.detach().cpu().numpy()))
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.loss_history.append(loss_val)
        return loss_val

    @torch.no_grad()
    def _soft_target_update(self):
        tau = self.target_tau
        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.mul_(1.0 - tau).add_(o_param, alpha=tau)

    def save(self, path: str):
        torch.save({
            "online_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_step": self.n_step,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "scheduler_step": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }, path)
        print(f"[DQN] 模型已保存至 {path}")

    def load(self, path: str):
        if path.lower().endswith(".json"):
            raise ValueError(
                f"{path} 是旧版 NumPy JSON 权重,和新的 PyTorch 网络不兼容。"
                "请改用 .pt 路径重新训练,例如 --model dqn_traffic_sumo.pt"
            )
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            data = torch.load(path, map_location=self.device)
        if data.get("state_dim") not in (None, self.state_dim):
            raise ValueError(
                f"模型状态维度 {data.get('state_dim')} 与当前 {self.state_dim} 不一致,请重新训练"
            )
        saved_hidden = data.get("hidden_dims")
        if saved_hidden is not None and tuple(saved_hidden) != tuple(self.hidden_dims):
            raise ValueError(
                f"模型隐藏层结构 {tuple(saved_hidden)} 与当前 {self.hidden_dims} 不一致,"
                "请使用相同结构加载或重新训练"
            )
        self.online_net.load_state_dict(data["online_state_dict"])
        self.target_net.load_state_dict(self.online_net.state_dict())
        if "optimizer_state_dict" in data:
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            # 学习率恢复策略:
            #  - 有调度器时,恢复调度进度,再由调度器按当前进度重设lr ——
            #    这样本次通过 lr=... 传入的新基准学习率依然生效(按进度缩放);
            #  - 无调度器时,直接把本次配置的lr写回每个参数组(旧逻辑)。
            #    原因: torch优化器的state_dict里连学习率本身也会被保存下来,
            #    直接 load_state_dict 会把学习率悄悄改回"上次保存模型时用的值",
            #    导致调用方通过 lr=... 传入的新学习率被静默忽略、完全不生效。
            #    无论哪种情况,都只保留 Adam 的动量/二阶矩估计
            #    (exp_avg/exp_avg_sq),学习率始终以本次运行传入的 lr 为准。
            saved_sched = data.get("scheduler_step")
            if self.lr_scheduler is not None:
                if isinstance(saved_sched, dict):
                    self.lr_scheduler.load_state_dict(saved_sched)
                # 旧checkpoint没有该字段时从进度0开始,等价于带着新超参重新warmup
                self.current_lr = self.lr_scheduler.apply()
            else:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.lr
                self.current_lr = self.lr
        saved_n_step = data.get("n_step")
        if saved_n_step is not None and int(saved_n_step) != self.n_step:
            print(f"[DQN] 警告: 模型按 n_step={int(saved_n_step)} 训练,当前配置 n_step={self.n_step},"
                  "回放池里新旧经验的回报视野不一致,建议先评估几轮确认效果,必要时重新训练")
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self._step_count = data.get("step_count", 0)
        print(f"[DQN] 模型已加载: {path}  ε={self.epsilon:.4f}")