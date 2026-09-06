# -*- coding: utf-8 -*-
"""
DQN 模块自测脚本 —— 不依赖 SUMO,几分钟内验证训练核心的正确性与学习能力。

覆盖:
    1. PER 优先经验回放: 新样本优先级取当前最大值(bug修复验证)、
       采样/权重/优先级更新的基本不变量
    2. n-step 经验回放: 折扣回报的手算值比对、done截断、episode结束flush
    3. 学习率调度器: warmup爬升、余弦退火、终点值、调度进度持久化
    4. save/load: 学习率以本次配置为准、epsilon恢复、旧格式checkpoint兼容
    5. 学习能力实测: 合成一维走廊MDP,对比"旧超参(lr=1e-3/MSE/1-step/硬更新)"
       与"新超参(lr=3e-3/Huber/3-step/软更新)"收敛到最优策略所需轮数

用法:
    python dqn_selftest.py            # 跑全部测试
    python dqn_selftest.py --quick    # 跳过学习能力对比(最快)
"""
import argparse
import os
import random
import sys
import tempfile

import numpy as np
import torch

from traffic_dqn import (
    TrafficDQNAgent,
    PrioritizedReplayBuffer,
    LinearWarmupCosineLR,
    Transition,
)

sys.stdout.reconfigure(encoding="utf-8")

PASS_COUNT = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  [通过] {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [失败] {name}" + (f"  ({detail})" if detail else ""))
        raise AssertionError(f"自测失败: {name} {detail}")


# ─────────────────────────────────────────────
# 1. PER 优先经验回放
# ─────────────────────────────────────────────
def test_per_buffer():
    print("\n== 测试 1: PER 优先经验回放 ==")
    buf = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)

    # 第一条经验: 树初始max优先级=1.0, push后树顶应为1.0
    t0 = Transition(np.zeros(4, dtype=np.float32), 0, 0.0,
                    np.zeros(4, dtype=np.float32), False)
    buf.push(t0)
    check("push后树顶优先级=1.0", abs(buf.total_priority - 1.0) < 1e-9,
          f"total={buf.total_priority:.6f}")

    # 用大TD误差更新优先级 → _max_pri=(|err|+eps)^alpha
    buf.update_priorities([buf.capacity - 1], np.array([10.0]))
    expect_max = (10.0 + buf.epsilon) ** buf.alpha
    check("update_priorities后_max_pri正确",
          abs(buf._max_pri - expect_max) < 1e-9,
          f"max={buf._max_pri:.6f}")

    # 关键bug修复验证: 新push的样本必须直接获得当前最大树内优先级,
    # 旧代码在这里又做了一次**alpha(alpha的平方),新样本优先级被压低
    t1 = Transition(np.ones(4, dtype=np.float32), 1, 1.0,
                    np.ones(4, dtype=np.float32), False)
    buf.push(t1)
    check("新样本优先级=当前最大值(bug修复)", abs(buf._max_pri - expect_max) < 1e-9,
          f"push后max仍为{buf._max_pri:.6f}")

    # 批量采样不变量: 数量正确、IS权重归一化、索引有效
    for _ in range(60):
        buf.push(Transition(np.random.rand(4).astype(np.float32), 0, 0.0,
                            np.random.rand(4).astype(np.float32), False))
    buf.update_priorities(
        [random.randrange(buf.capacity - 1, buf.capacity - 1 + buf._size) for _ in range(60)],
        np.random.rand(60) * 5,
    )
    trans, indices, weights = buf.sample(32)
    check("采样数量=batch_size", len(trans) == 32 and len(indices) == 32)
    check("IS权重全部>0且最大值=1",
          float(weights.min()) > 0 and abs(float(weights.max()) - 1.0) < 1e-6)
    check("所有采样转移非None且状态维度=4",
          all(t is not None and t.state.shape == (4,) for t in trans))

    # 优先级更新后: 树内部节点求和仍与叶子一致, _max_pri只增不减
    max_before = buf._max_pri
    buf.update_priorities(indices, np.abs(np.random.randn(32)) + 0.1)
    leaf_sum = buf._tree[buf.capacity - 1:buf.capacity - 1 + buf._size].sum()
    check("更新优先级后SumTree内部求和=叶子求和",
          abs(leaf_sum - buf.total_priority) < 1e-6 * max(1.0, leaf_sum))
    check("更新优先级后_max_pri不减小", buf._max_pri >= max_before - 1e-12)
    trans, indices, weights = buf.sample(32)
    check("更新优先级后采样仍正常", len(trans) == 32 and float(weights.max()) <= 1.0 + 1e-6)


# ─────────────────────────────────────────────
# 2. n-step 经验回放
# ─────────────────────────────────────────────
def test_nstep_store():
    print("\n== 测试 2: n-step 经验聚合 ==")
    agent = TrafficDQNAgent(state_dim=3, action_dim=2, n_step=3, gamma=0.5,
                            lr_warmup_steps=0, lr_total_steps=0)
    s = lambda i: np.array([i, 0, 0], dtype=np.float32)

    # 5条经验 r=1..5, 最后一条done; 手算期望的n-step回报
    for i, r in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        agent.store(s(i), i % 2, r, s(i + 1), done=(i == 4))

    check("5条原始经验(n=3)应聚合成3条回放经验", len(agent.replay) == 3,
          f"实际{len(agent.replay)}条")
    stored = [agent.replay._data[i] for i in range(3)]
    g = agent.gamma
    # 窗口1: r1 + γr2 + γ²r3, bootstrap到s3
    check("第1条回报=1+0.5*2+0.25*3=2.75",
          abs(stored[0].reward - (1 + g * 2 + g * g * 3)) < 1e-6,
          f"实际{stored[0].reward:.4f}")
    check("第1条next_state=s3", np.allclose(stored[0].next_state, s(3)))
    check("第1条done=False", stored[0].done is False)
    # 窗口2: r2 + γr3 + γ²r4
    check("第2条回报=2+0.5*3+0.25*4=4.5",
          abs(stored[1].reward - (2 + g * 3 + g * g * 4)) < 1e-6,
          f"实际{stored[1].reward:.4f}")
    # 窗口3(done截断): r3 + γr4 + γ²r5, done=True, next_state=终态s5
    check("第3条回报=3+0.5*4+0.25*5=6.25(done截断)",
          abs(stored[2].reward - (3 + g * 4 + g * g * 5)) < 1e-6,
          f"实际{stored[2].reward:.4f}")
    check("第3条done=True且next_state=终态s5",
          stored[2].done is True and np.allclose(stored[2].next_state, s(5)))

    # 不足n条就done: 2条也应正确聚合并flush
    agent2 = TrafficDQNAgent(state_dim=3, action_dim=2, n_step=3, gamma=0.5,
                             lr_warmup_steps=0, lr_total_steps=0)
    agent2.store(s(0), 0, 1.0, s(1), done=False)
    agent2.store(s(1), 0, 2.0, s(2), done=True)
    check("2条就done(不足n=3)也应产出1条经验", len(agent2.replay) == 1)
    only = agent2.replay._data[0]
    check("截断回报=1+0.5*2=2.0", abs(only.reward - 2.0) < 1e-6)
    check("截断经验done=True", only.done is True)

    # flush_replay: 未触发done时手动冲入尾巴(对应SUMO每轮max_sim_time强制结束)
    agent3 = TrafficDQNAgent(state_dim=3, action_dim=2, n_step=3, gamma=0.5,
                             lr_warmup_steps=0, lr_total_steps=0)
    agent3.store(s(0), 0, 1.0, s(1), done=False)
    agent3.store(s(1), 0, 2.0, s(2), done=False)
    check("不足n条时flush前不入池", len(agent3.replay) == 0)
    agent3.flush_replay()
    check("flush_replay后尾巴入池", len(agent3.replay) == 1)
    check("flush回报=1+0.5*2=2.0(无done不折扣bootstrap)",
          abs(agent3.replay._data[0].reward - 2.0) < 1e-6,
          f"实际{agent3.replay._data[0].reward:.4f}")


# ─────────────────────────────────────────────
# 3. 学习率调度器
# ─────────────────────────────────────────────
def test_lr_scheduler():
    print("\n== 测试 3: 学习率调度器(warmup + 余弦退火) ==")
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1.0)
    sched = LinearWarmupCosineLR(opt, base_lr=3e-3, warmup_steps=10,
                                 total_steps=100, min_lr_scale=0.1)
    check("warmup起点≈base_lr/10", abs(sched.lr_at(0) - 3e-4) < 1e-9,
          f"lr={sched.lr_at(0):.2e}")
    check("warmup终点=base_lr", abs(sched.lr_at(9) - 3e-3) < 1e-9)
    check("退火中点=base_lr与min的中点",
          abs(sched.lr_at(55) - (3e-4 + (3e-3 - 3e-4) * 0.5)) < 1e-9,
          f"lr={sched.lr_at(55):.2e}")
    check("退火终点=min_lr", abs(sched.lr_at(100) - 3e-4) < 1e-9)

    lrs = [sched.lr_at(i) for i in range(10, 101)]
    check("warmup后单调不增", all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1)))

    # apply把lr写进optimizer; step推进进度
    sched.step()  # _step=1
    cur = sched.apply()
    check("apply将调度lr写入param_groups",
          abs(opt.param_groups[0]["lr"] - cur) < 1e-12,
          f"lr={cur:.2e}")
    sd = sched.state_dict()
    sched2 = LinearWarmupCosineLR(opt, base_lr=3e-3, warmup_steps=10, total_steps=100)
    sched2.load_state_dict(sd)
    check("调度进度可save/load还原", sched2._step == sched._step)

    # 只warmup不退火(total_steps=0)
    sched3 = LinearWarmupCosineLR(opt, base_lr=3e-3, warmup_steps=10, total_steps=0)
    check("total_steps=0时warmup后保持base_lr",
          abs(sched3.lr_at(9999) - 3e-3) < 1e-12)


# ─────────────────────────────────────────────
# 4. save / load
# ─────────────────────────────────────────────
def test_save_load():
    print("\n== 测试 4: 模型保存/加载 ==")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.pt")

    agent_a = TrafficDQNAgent(state_dim=6, action_dim=3, lr=3e-3,
                              lr_warmup_steps=10, lr_total_steps=1000,
                              min_replay_size=64)
    for _ in range(80):
        agent_a.store(np.random.rand(6).astype(np.float32), 0, 0.5,
                      np.random.rand(6).astype(np.float32), False)
        agent_a.train_step()
    agent_a.epsilon = 0.42
    agent_a.save(path)
    steps_a = agent_a._step_count
    check("保存后调度进度>0", agent_a.lr_scheduler._step == steps_a,
          f"sched_step={agent_a.lr_scheduler._step}")

    agent_b = TrafficDQNAgent(state_dim=6, action_dim=3, lr=5e-3,
                              lr_warmup_steps=10, lr_total_steps=1000,
                              min_replay_size=64)
    agent_b.load(path)
    # 有调度器: lr按恢复的调度进度、以本次base_lr=5e-3重新缩放
    lr_b = agent_b.optimizer.param_groups[0]["lr"]
    expect = agent_b.lr_scheduler.lr_at(steps_a)
    check("加载后lr=新base_lr按调度进度缩放(而非旧checkpoint里的lr)",
          abs(lr_b - expect) < 1e-12, f"lr={lr_b:.2e} 期望{expect:.2e}")
    check("epsilon从checkpoint恢复", abs(agent_b.epsilon - 0.42) < 1e-9)
    same = all(torch.allclose(p1, p2) for p1, p2 in
               zip(agent_a.online_net.parameters(), agent_b.online_net.parameters()))
    check("网络权重完整还原", same)

    # 无调度器(lr_total_steps=0且warmup=0): lr强制等于本次配置值
    agent_c = TrafficDQNAgent(state_dim=6, action_dim=3, lr=3e-3,
                              lr_warmup_steps=0, lr_total_steps=0,
                              min_replay_size=64)
    agent_c.save(path)
    agent_d = TrafficDQNAgent(state_dim=6, action_dim=3, lr=7e-3,
                              lr_warmup_steps=0, lr_total_steps=0,
                              min_replay_size=64)
    agent_d.load(path)
    check("无调度器时加载后lr=本次传入的lr",
          abs(agent_d.optimizer.param_groups[0]["lr"] - 7e-3) < 1e-12)

    # 旧格式checkpoint(只有旧版字段)也能加载
    legacy = os.path.join(tmpdir, "legacy.pt")
    torch.save({
        "online_state_dict": agent_c.online_net.state_dict(),
        "optimizer_state_dict": agent_c.optimizer.state_dict(),
        "epsilon": 0.3,
        "step_count": 10,
        "state_dim": 6,
        "action_dim": 3,
    }, legacy)
    agent_e = TrafficDQNAgent(state_dim=6, action_dim=3, lr=3e-3,
                              lr_warmup_steps=10, lr_total_steps=1000,
                              min_replay_size=64)
    agent_e.load(legacy)
    check("旧格式checkpoint兼容加载", abs(agent_e.epsilon - 0.3) < 1e-9)


# ─────────────────────────────────────────────
# 5. 学习能力实测: 一维走廊MDP
# ─────────────────────────────────────────────
CORRIDOR_N = 10          # 状态0..9, 9为终点
CORRIDOR_MAX_STEPS = 25


def _corridor_run(agent: TrafficDQNAgent, seed: int, max_episodes: int):
    """跑走廊MDP,返回首次达到最优贪心策略所需的episode数(达不到返回None)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent.epsilon = 1.0
    best_ep = None
    for ep in range(1, max_episodes + 1):
        pos = random.randrange(CORRIDOR_N - 1)   # 起点0..8
        state = _onehot(pos)
        for _ in range(CORRIDOR_MAX_STEPS):
            action = agent.select_action(state)
            if action == 1:                       # right
                npos = pos + 1
                reward, nstate, done = (-0.01, _onehot(npos), npos == CORRIDOR_N - 1)
            else:                                 # left
                npos = max(pos - 1, 0)
                reward, nstate, done = (-0.01, _onehot(npos), False)
            agent.store(state, action, reward, nstate, done)
            agent.train_step()
            state, pos = nstate, npos
            if done:
                break
        else:
            agent.flush_replay()                  # 达到步数上限,冲入尾巴经验
        if ep % 2 == 0 and _corridor_policy_optimal(agent):
            best_ep = best_ep or ep
    return best_ep, ep


def _onehot(pos: int) -> np.ndarray:
    v = np.zeros(CORRIDOR_N, dtype=np.float32)
    v[pos] = 1.0
    return v


def _corridor_policy_optimal(agent: TrafficDQNAgent) -> bool:
    """贪心策略在每个非终点状态都应选'右'"""
    for pos in range(CORRIDOR_N - 1):
        if agent.select_action(_onehot(pos), greedy=True) != 1:
            return False
    return True


def test_learning_capability(quick: bool):
    print("\n== 测试 5: 学习能力实测(一维走廊MDP, 3个随机种子) ==")
    configs = {
        # 旧版默认: 恒定学习率1e-3、纯MSE、1-step、每200步硬拷贝、batch 64
        "旧超参 lr=1e-3/MSE/1-step/硬更新": dict(lr=1e-3, use_huber=False, n_step=1,
                                                target_tau=0.0, batch_size=64,
                                                lr_warmup_steps=0, lr_total_steps=0),
        # 新版默认: 3e-3 + warmup/退火、Huber、3-step、软更新、batch 128
        "新超参 lr=3e-3/Huber/3-step/软更新": dict(lr=3e-3, use_huber=True, n_step=3,
                                                target_tau=0.005, batch_size=128,
                                                lr_warmup_steps=50, lr_total_steps=5000),
    }
    max_episodes = 25 if quick else 80
    results = {}
    for name, kwargs in configs.items():
        eps_needed = []
        for seed in (7, 42, 2026):
            agent = TrafficDQNAgent(state_dim=CORRIDOR_N, action_dim=2, gamma=0.95,
                                    min_replay_size=64, epsilon_decay=0.99,
                                    lr_min_scale=0.1, **kwargs)
            best_ep, _ = _corridor_run(agent, seed, max_episodes)
            eps_needed.append(best_ep)
        results[name] = eps_needed
        fmt = ", ".join("未收敛" if e is None else f"ep{e}" for e in eps_needed)
        print(f"  {name}: 首次最优episode = [{fmt}]")

    new_vals = results["新超参 lr=3e-3/Huber/3-step/软更新"]
    check("新超参在全部种子上收敛到最优策略", all(v is not None for v in new_vals),
          f"结果={new_vals}")
    old_vals = results["旧超参 lr=1e-3/MSE/1-step/硬更新"]
    new_avg = np.mean([v for v in new_vals if v is not None])
    old_avg = (np.mean([v for v in old_vals if v is not None])
               if all(v is not None for v in old_vals) else float("inf"))
    print(f"  平均收敛速度: 新={new_avg:.0f}ep  旧={old_avg:.0f}ep  "
          f"(新比旧快{old_avg / max(new_avg, 1e-9):.1f}x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="跳过学习能力对比")
    args = parser.parse_args()

    print("=" * 60)
    print("DQN 模块自测开始 (torch", torch.__version__, ")")
    print("=" * 60)

    test_per_buffer()
    test_nstep_store()
    test_lr_scheduler()
    test_save_load()
    test_learning_capability(quick=args.quick)

    print("\n" + "=" * 60)
    print(f"全部自测通过, 共 {PASS_COUNT} 项检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
