# -*- coding: utf-8 -*-
"""
SUMO 版主管线 —— 替代 pipeline.py 里"视频帧 → YOLO → DeepSORT"这一段,
直接用 TraCI 驱动 SUMO 仿真,状态和信号控制都基于仿真真值。

traffic_dqn.py 里的 TrafficDQNAgent 完全不用改,直接复用。

新版训练超参(相对旧版的提升):
    · 学习率默认 3e-3(旧1e-3),带线性warmup+余弦退火,收敛更快且不发散
    · 批量 128(旧64)、min-replay 512(旧256),初始梯度更稳
    · Huber损失 + 目标网络软更新(τ=0.005),高学习率下的双保险
    · 3-step 经验回放,奖励信号更快回传,信用分配更快
    · 修复 PER 新样本优先级被二次幂压低的bug
    · epsilon 衰减放缓(0.997),多留几轮探索,避免过早收敛到偏科策略

用法:
    # 默认接入项目内 sumo/ (来自 Desktop/sumo_app(1)),跑路口1
    python sumo_pipeline.py

    # 多轮训练(推荐): 连续跑50轮,每轮独立随机车流种子
    python sumo_pipeline.py --episodes 50 --model dqn_traffic_sumo.pt

    # 对比原始配时 / Webster / 爬山延误
    python delay_score.py --sumocfg sumo/intersection_1.sumocfg --hill-climb

    # 换路口(只需给 cfg 名,net 会从 sumocfg 里自动读)
    python sumo_pipeline.py --sumocfg intersection_2.sumocfg

    # 多路口训练一个通用模型: 'all'=全部20个路口, 每轮轮换一个,
    # reward尺度校准按路口分别维护; 评估模式则逐个路口测评
    python sumo_pipeline.py --sumocfg all --episodes 200 --model dqn_generalist.pt

    # 评估模式(不训练,加载已有模型,eps=0)
    python sumo_pipeline.py --eval --gui
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time

SUMO_HOME = os.environ.get("SUMO_HOME", r"D:\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402

from sumo_state_builder import SumoStateBuilder, PHASE_NAMES  # noqa: E402
from sumo_traffic_env import SumoTrafficEnv  # noqa: E402
from delay_score import mean_waiting_from_tripinfo  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
from traffic_dqn import TrafficDQNAgent  # noqa: E402  (来自原项目,直接复用)

DEFAULT_SUMO_ROOT = os.path.join(PROJECT_DIR, "sumo")
FALLBACK_SUMO_ROOT = r"C:\Users\86187\Desktop\sumo_app(1)"


def get_sumo_root() -> str:
    if os.path.isfile(os.path.join(DEFAULT_SUMO_ROOT, "intersection_1.sumocfg")):
        return DEFAULT_SUMO_ROOT
    return FALLBACK_SUMO_ROOT


def parse_net_from_sumocfg(sumocfg: str) -> str:
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg))
    with open(sumocfg, encoding="utf-8") as f:
        text = f.read()
    match = re.search(r'<net-file\s+value="([^"]+)"', text)
    if match:
        return os.path.normpath(os.path.join(cfg_dir, match.group(1)))
    name = os.path.splitext(os.path.basename(sumocfg))[0]
    return os.path.join(cfg_dir, "net", f"{name}.net.xml")


def resolve_scenario_paths(args):
    """把 --sumocfg / --net 解析成 (sumocfg, net) 场景列表, 存入 args.scenarios。
    --sumocfg 支持: 单个名称/路径 | 逗号分隔多个 | 'all'(sumo-root下全部
    intersection_*.sumocfg)。多场景训练要求所有路口状态维度一致(main里校验)。"""
    args.sumo_root = os.path.abspath(args.sumo_root)
    if not args.sumocfg:
        raw = ["intersection_1.sumocfg"]
    elif args.sumocfg.strip().lower() == "all":
        raw = sorted(
            f for f in os.listdir(args.sumo_root)
            if (f.startswith("intersection_") or f.startswith("arterial_"))
            and f.endswith(".sumocfg"))
        if not raw:
            raise FileNotFoundError(f"{args.sumo_root} 下没有可用的 sumocfg")
    else:
        raw = [s.strip() for s in args.sumocfg.split(",") if s.strip()]

    scenarios = []
    for name in raw:
        sumocfg = name if os.path.isfile(name) else os.path.join(args.sumo_root, name)
        sumocfg = os.path.abspath(sumocfg)
        if not os.path.isfile(sumocfg):
            raise FileNotFoundError(f"找不到 sumocfg: {sumocfg}")
        # 多场景时 net 一律从各 sumocfg 读取; --net 只在单场景时允许覆盖
        net = parse_net_from_sumocfg(sumocfg)
        if len(raw) == 1 and args.net:
            net = args.net if os.path.isfile(args.net) else os.path.join(args.sumo_root, args.net)
        net = os.path.abspath(net)
        if not os.path.isfile(net):
            raise FileNotFoundError(f"找不到 net.xml: {net}")
        scenarios.append((sumocfg, net))
    args.scenarios = scenarios
    args.sumocfg, args.net = scenarios[0]   # 兼容旧代码对单场景字段的引用


def parse_args():
    p = argparse.ArgumentParser(description="SUMO 版交通信号 DQN 控制管线")
    p.add_argument("--sumo-root", default=get_sumo_root(),
                   help="SUMO 场景根目录,默认项目内 sumo/ (由 sumo_app(1) 接入)")
    p.add_argument("--sumocfg", default=None,
                   help="sumocfg 路径/文件名; 逗号分隔可训多个路口(如 'intersection_1.sumocfg,intersection_18.sumocfg');"
                        " 'all' 表示 sumo-root 下全部 intersection_*.sumocfg(通用模型训练)。"
                        " 默认 intersection_1.sumocfg")
    p.add_argument("--net", default=None,
                   help="net.xml 路径;不填则从 sumocfg 的 <net-file> 自动读取")
    p.add_argument("--tls-id", default=None,
                   help="信号灯ID, 默认None=自动接管路网内全部信号灯(单路口仅1个,"
                        "干线4~5个全部协同)。单路口旧用法--tls-id J仍兼容")
    p.add_argument("--model", default="dqn_traffic_sumo.pt", help="DQN模型保存/加载路径(.pt)")
    p.add_argument("--eval", action="store_true", help="评估模式(不训练, epsilon=0)")
    p.add_argument("--min-green", default=10, type=int, help="最小绿灯秒数,同时是一次决策保持的绿灯时长")
    p.add_argument("--max-green", default=60, type=int, help="最大绿灯秒数")
    p.add_argument("--min-replay-size", default=512, type=int,
                   help="开始训练前至少攒够多少条经验(默认512,batch=128时初始梯度更稳)")
    p.add_argument("--epsilon-decay", default=0.997, type=float,
                   help="epsilon每次train_step衰减的乘数(默认0.997,比旧值0.995衰减更慢,"
                        "多留2~3轮充分探索,避免过早收敛到偏科策略)")
    p.add_argument("--max-sim-time", default=3600, type=float,
                   help="单轮episode最长仿真秒数,建议和sumocfg里<end value>一致")
    p.add_argument("--reward-alpha", default=5.0, type=float,
                   help="奖励函数:等待时间变化率项的最大贡献量级(默认5.0)。"
                        "新版reward用tanh做了平滑饱和处理,这个值代表该项"
                        "封顶能贡献多少reward,不再是纯线性斜率")
    p.add_argument("--reward-delay-coef", default=5.0, type=float,
                   help="奖励函数:绝对人均等待时间惩罚项的最大贡献量级(默认5.0,"
                        "配合tanh饱和,不管路口拥堵到多严重都不会让这一项无限增大)")
    p.add_argument("--reward-gamma", default=2.0, type=float,
                   help="奖励函数:吞吐量奖励权重(线性项,未做tanh处理,"
                        "因为throughput_per_sec本身量级天然有限)")
    p.add_argument("--reward-clip", default=30.0, type=float,
                   help="奖励裁剪范围,即reward被限制在[-此值,+此值]之间(默认30)。"
                        "改用tanh饱和后各项理论上已经有界,这个只是最后一道安全网,"
                        "正常情况下应该很少真正触发裁剪")
    p.add_argument("--reward-fairness-coef", default=5.0, type=float,
                   help="公平性惩罚项的最大贡献量级(默认5.0),针对等待最久的那条车道"
                        "额外扣分,防止智能体学出偏袒某个方向、长期饿着另一个方向的"
                        "偷懒策略(实测中不加这项会出现某方向绿灯占比从~25%涨到~50%、"
                        "另一方向被压到不足10%、车均延误反而越训练越差的问题)")
    p.add_argument("--reward-wait-scale", default=60.0, type=float,
                   help="reward里tanh饱和曲线的等待时间参考尺度,单位:秒(默认60)。"
                        "控制'车均等待时间'和'最长车道等待时间'这两项大概在多少秒"
                        "开始饱和。如果发现reward长期贴着上限或下限不变化(说明还是"
                        "饱和了、丢失了区分度),可以调大这个值,让曲线更晚饱和,"
                        "拉开中间区间的区分度(路口越拥堵,可能需要调得越大)")
    p.add_argument("--reward-delta-scale", default=5.0, type=float,
                   help="reward里tanh饱和曲线的'等待时间变化率'参考尺度(默认5,"
                        "即每秒等待时间上涨5秒左右这一项就接近饱和)。同样,"
                        "如果发现这一项长期饱和无区分度,可以调大")
    p.add_argument("--lr", default=3e-3, type=float,
                   help="DQN学习率基准值(已从旧默认1e-3提高到3e-3,配合warmup+余弦退火:"
                        "冷启动先线性warmup到3e-3快速学习,再按余弦曲线退火到其10%%,"
                        "前期快、后期稳,不会因为学习率高而发散)。"
                        "如仍观察到avg_loss持续上涨不收敛,可降到1e-3或配合"
                        "--target-update-tau调小")
    p.add_argument("--lr-warmup", default=500, type=int,
                   help="学习率线性warmup步数(默认500,前500个训练步从接近0升到--lr,"
                        "避免冷启动时被大学习率猛踢;设0关闭warmup)")
    p.add_argument("--lr-total-steps", default=50000, type=int,
                   help="余弦退火总步数(默认50000,约对应175轮episode;到该步数时学习率"
                        "退火到--lr*--lr-min-scale。设0则不退火,始终保持--lr)")
    p.add_argument("--lr-min-scale", default=0.1, type=float,
                   help="退火终点学习率=--lr*此系数(默认0.1,即后期降到3e-4精细收敛)")
    p.add_argument("--target-update-tau", default=0.005, type=float,
                   help="目标网络Polyak软更新系数(默认0.005,每个训练步轻微拉动目标网络,"
                        "比每200步硬拷贝更平滑,高学习率下目标Q值抖动更小;设0退回硬同步,"
                        "此时同步频率由--target-update-freq控制)")
    p.add_argument("--n-step", default=3, type=int,
                   help="n-step经验回放的视野步数(默认3):一次决策约10秒绿灯,3步约30秒,"
                        "奖励信号能更快传播到更早决策,显著加快信用分配;设1退化为普通1-step")
    p.add_argument("--loss", default="huber", choices=["huber", "mse"],
                   help="训练损失类型(默认huber):huber对离群TD误差的梯度有上界,"
                        "是高学习率下的关键稳定器; mse为旧版纯平方误差,供对比实验")
    p.add_argument("--gamma", default=0.95, type=float,
                   help="DQN折扣因子(默认0.95)。值越大代表越看重长期回报,"
                        "但配合较大学习率时也更容易导致Q值估计不稳定,"
                        "如果loss发散可以尝试调低到0.9")
    p.add_argument("--target-update-freq", default=200, type=int,
                   help="目标网络同步频率,单位:训练步数(默认200,即TrafficDQNAgent默认值)。"
                        "值越小,目标网络更新越频繁,训练目标越不稳定,容易导致loss发散不收敛;"
                        "如果观察到这个现象,可以尝试调大到500~1000让目标网络更稳定")
    p.add_argument("--batch-size", default=128, type=int,
                   help="DQN训练时每次采样的批量大小(已从64提高到128,梯度噪声更小,"
                        "支撑更高的学习率;SUMO仿真本身比训练慢得多,训练开销可忽略)")
    p.add_argument("--gui", action="store_true", help="用sumo-gui可视化运行(调试用)")
    p.add_argument("--log-every", default=20, type=int, help="每隔多少个决策步打印一次日志")
    p.add_argument("--episodes", default=1, type=int,
                   help="训练轮数(episode数)。每轮都是一次完整的--max-sim-time秒仿真,"
                        "轮与轮之间DQN的权重和经验回放会保留、持续训练,只是SUMO仿真"
                        "本身会重新开始跑。eval模式下这个参数也生效,可以用来看多轮平均表现。")
    p.add_argument("--seed-start", default=0, type=int,
                   help="第1轮用的SUMO随机种子,之后每轮自动+1(--seed-start 0 -> 第1轮种子0,第2轮种子1...)。"
                        "不同种子=不同的车辆生成随机过程,让DQN见到更多样的路况,不是每轮都一模一样。")
    p.add_argument("--episode-log", default="episode_log.csv",
                   help="每轮汇总统计写入的CSV文件路径,可以后续画图看训练有没有在变好")
    p.add_argument("--collect-data", action="store_true",
                   help="启用功能二数据采集(车辆轨迹/进口道排队/相位/油耗逐秒记录到"
                        "collected_data/, 供dashboard实时仪表盘与评估报告使用)")
    return p.parse_args()


def eps_sidecar_path(model_path: str) -> str:
    """epsilon 存在模型文件旁边的一个小文本文件里,例如 dqn_traffic_sumo.pt.eps.txt"""
    return f"{model_path}.eps.txt"


def save_epsilon(model_path: str, epsilon: float):
    try:
        with open(eps_sidecar_path(model_path), "w", encoding="utf-8") as f:
            f.write(str(epsilon))
    except OSError as exc:
        print(f"[SUMO Pipeline] 警告: epsilon 边车文件保存失败: {exc}")


def load_epsilon(model_path: str):
    eps_path = eps_sidecar_path(model_path)
    if not os.path.exists(eps_path):
        return None
    try:
        with open(eps_path, encoding="utf-8") as f:
            return float(f.read().strip())
    except (OSError, ValueError) as exc:
        print(f"[SUMO Pipeline] 警告: epsilon 边车文件读取失败,将忽略: {exc}")
        return None


def calib_sidecar_path(model_path: str) -> str:
    """reward尺度校准值存在模型文件旁边,例如 dqn_traffic_sumo.pt.calib.json"""
    return f"{model_path}.calib.json"


def save_calib(model_path: str, calib: dict):
    """保存各路口的reward尺度校准值。calib: {场景tag: (wait_scale, delta_scale)}。
    多路口训练时各路口拥堵程度不同,必须分别校准,不能共用一套尺度。"""
    try:
        data = {"per_scenario": {tag: {"wait_scale": w, "delta_scale": d}
                                 for tag, (w, d) in calib.items()}}
        with open(calib_sidecar_path(model_path), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as exc:
        print(f"[SUMO Pipeline] 警告: reward校准边车文件保存失败: {exc}")


def load_calib(model_path: str):
    """读取各路口的校准值,返回 {场景tag: (wait_scale, delta_scale)} 或 None。
    兼容旧版扁平格式 {"wait_scale":..,"delta_scale":..}(映射到 '_default')。"""
    path = calib_sidecar_path(model_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[SUMO Pipeline] 警告: reward校准边车文件读取失败,将忽略: {exc}")
        return None
    if "per_scenario" in data:
        return {tag: (float(v["wait_scale"]), float(v["delta_scale"]))
                for tag, v in data["per_scenario"].items()}
    if "wait_scale" in data:
        return {"_default": (float(data["wait_scale"]), float(data["delta_scale"]))}
    return None


def get_resume_state(log_path: str):
    """
    断点续训: 如果 episode_log 已经存在(说明之前训练过至少一轮),
    读出最后一行记录的 episode 和 seed,这样重新运行时可以自动接着往下训,
    而不是每次都从 episode 1 / seed 0 重新开始。
    返回 (上次最后完成的episode编号, 上次最后用的seed) 或 None(还没训练过)。
    """
    if not os.path.exists(log_path):
        return None
    last_episode, last_seed = None, None
    try:
        with open(log_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_episode = int(row["episode"])
                    last_seed = int(row["seed"])
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        return None
    if last_episode is None:
        return None
    return last_episode, last_seed


def get_sumo_binary(gui: bool) -> str:
    sumo_exe_name = "sumo-gui.exe" if gui else "sumo.exe"
    sumo_binary = os.path.join(SUMO_HOME, "bin", sumo_exe_name)
    if not os.path.exists(sumo_binary):
        sumo_binary = "sumo-gui" if gui else "sumo"
    return sumo_binary


def start_sumo(sumocfg: str, seed: int, gui: bool, tripinfo_path: str = None, label: str = None):
    """在 sumocfg 所在目录启动 SUMO,这样 cfg 里的相对路径(net/routes/results)都能对上。
    label: TraCI连接标签——平台多线程场景(信号仿真/评估并行)用独立标签隔离连接,
    不传则使用默认连接(兼容旧行为)。"""
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg))
    os.makedirs(os.path.join(cfg_dir, "results"), exist_ok=True)
    cmd = [get_sumo_binary(gui), "-c", os.path.abspath(sumocfg),
           "--no-warnings", "--seed", str(seed)]
    if tripinfo_path:
        os.makedirs(os.path.dirname(os.path.abspath(tripinfo_path)), exist_ok=True)
        cmd.extend(["--tripinfo-output", os.path.abspath(tripinfo_path)])
    if gui:
        cmd.append("--start")
    old_cwd = os.getcwd()
    os.chdir(cfg_dir)
    try:
        if label:
            traci.start(cmd, label=label)
        else:
            traci.start(cmd)
    finally:
        os.chdir(old_cwd)


def run_episode(args, agent, state_builder_cls, env_cls, episode_idx: int, seed: int, train_mode: bool,
                 sumocfg: str, net: str,
                 calib_wait_scale: float = None, calib_delta_scale: float = None):
    """
    跑一整轮(一次完整的SUMO仿真,从0秒到--max-sim-time)。
    agent 是在外层创建、跨轮次共享的,权重和经验回放池不会因为换轮次重置。
    sumocfg/net: 本轮使用的路口场景(多场景训练时每轮轮换一个路口)。
    calib_wait_scale/calib_delta_scale: 该路口跨episode慢慢校准出来的reward尺度
    参考值(见 sumo_traffic_env.SumoTrafficEnv 的说明),本episode内固定不变。
    返回这一轮的汇总统计字典,用来写进 episode_log.csv,
    其中额外包含 observed_wait_scale/observed_delta_scale,
    供外层用来更新下一轮要用的校准值。
    """
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg))
    cfg_tag = os.path.splitext(os.path.basename(sumocfg))[0]
    tripinfo_path = os.path.join(
        cfg_dir, "results", f"tripinfo_dqn_{cfg_tag}_ep{episode_idx}_seed{seed}.xml")
    start_sumo(sumocfg, seed, args.gui, tripinfo_path=tripinfo_path)

    # 功能二数据采集: --collect-data 时把采集器挂到每个仿真步上
    # (惰性导入: data_collector 反向复用了本文件的 start_sumo, 顶层导入会循环依赖)
    collector = None
    if getattr(args, "collect_data", False):
        from data_collector import SimulationDataCollector
        collector = SimulationDataCollector(
            traci, net, os.path.join("collected_data", f"{cfg_tag}_ep{episode_idx}_seed{seed}"),
            label=f"{cfg_tag}_ep{episode_idx}_seed{seed}",
            control_mode="dqn" if train_mode else "dqn_eval",
            scenario=cfg_tag, seed=seed)

    t0 = time.time()
    try:
        state_builder = state_builder_cls(traci, net, tls_id=args.tls_id)
        env = env_cls(
            traci, net, tls_id=args.tls_id,
            min_green_sec=args.min_green, max_green_sec=args.max_green,
            max_sim_time=args.max_sim_time,
            reward_alpha=args.reward_alpha, reward_delay_coef=args.reward_delay_coef,
            reward_gamma=args.reward_gamma, reward_clip=args.reward_clip,
            reward_fairness_coef=args.reward_fairness_coef,
            reward_wait_scale=args.reward_wait_scale, reward_delta_scale=args.reward_delta_scale,
            calib_wait_scale=calib_wait_scale, calib_delta_scale=calib_delta_scale,
            step_callback=collector.on_sim_step if collector else None,
        )
        env.reset()

        def build_all_states():
            """逐路口构建状态(含邻居相位上下文, 干线协同的关键输入)"""
            tls_phases = {t: (env.current_phase[t], env.phase_duration_sec[t])
                          for t in env.tls_ids}
            return {t: state_builder.build(t, env.current_phase[t],
                                           env.phase_duration_sec[t], tls_phases)
                    for t in env.tls_ids}

        current_states = build_all_states()

        decision_step = 0
        reward_sum, raw_reward_sum, loss_sum, loss_count = 0.0, 0.0, 0.0, 0
        phase_counts = {a: 0 for a in PHASE_NAMES}

        while not env.is_finished():
            actions = {}
            for t in env.tls_ids:
                a = agent.select_action(current_states[t], greedy=not train_mode)
                actions[t] = env.clamp_action(t, a)
            env.apply_actions(actions)
            next_states = build_all_states()
            rewards = env.compute_reward()          # 每路口一条 [(clipped, raw), ...]
            rewards_map = {t: rewards[i] for i, t in enumerate(env.tls_ids)}
            step_reward = sum(r for r, _ in rewards) / len(env.tls_ids)
            step_raw = sum(r for _, r in rewards) / len(env.tls_ids)

            if train_mode:
                done = env.is_finished()
                for t in env.tls_ids:
                    agent.store(current_states[t], actions[t],
                                rewards_map[t][0], next_states[t], done)
                loss = agent.train_step()
                if loss is not None:
                    loss_sum += loss
                    loss_count += 1
            else:
                loss = None

            current_states = next_states
            decision_step += 1
            reward_sum += step_reward
            raw_reward_sum += step_raw
            for t in env.tls_ids:
                phase_counts[env.current_phase[t]] += 1

            if decision_step % args.log_every == 0:
                sim_time = traci.simulation.getTime()
                msg = (f"  [ep{episode_idx} t={sim_time:6.0f}s] decision#{decision_step} "
                       f"phase={'/'.join(str(env.current_phase[t]) for t in env.tls_ids)} "
                       f"reward={step_reward:+.3f}(raw={step_raw:+.3f}) "
                       f"eps={agent.epsilon:.4f} lr={agent.current_lr:.2e}")
                if loss is not None:
                    msg += f" loss={loss:.4f}"
                print(msg)

        # episode结束时把n-step滑动窗口里剩余的经验也冲进回放池,
        # 否则每轮最后 n-1 条决策的经验会一直滞留在窗口里丢失掉。
        if train_mode:
            agent.flush_replay()

        wall_clock_sec = time.time() - t0
        observed_wait_scale, observed_delta_scale = env.get_episode_scale_observations()
    finally:
        if collector is not None:
            collector.close()
        traci.close()

    mean_delay = mean_waiting_from_tripinfo(tripinfo_path)
    return {
        "episode": episode_idx,
        "seed": seed,
        "scenario": cfg_tag,
        "decisions": decision_step,
        "avg_reward": reward_sum / max(decision_step, 1),
        "avg_raw_reward": raw_reward_sum / max(decision_step, 1),
        "avg_loss": (loss_sum / loss_count) if loss_count else float("nan"),
        "epsilon_end": agent.epsilon,
        "phase_counts": phase_counts,
        "wall_clock_sec": wall_clock_sec,
        "mean_delay": mean_delay,
        "tripinfo_path": tripinfo_path,
        "observed_wait_scale": observed_wait_scale,
        "observed_delta_scale": observed_delta_scale,
    }


def append_episode_log(log_path: str, stats: dict):
    fieldnames = ["episode", "seed", "scenario", "decisions", "avg_reward", "avg_raw_reward",
                  "avg_loss", "epsilon_end", "wall_clock_sec", "mean_delay"] + \
                 [f"phase_{a}_count" for a in PHASE_NAMES]
    row = {k: stats[k] for k in ["episode", "seed", "scenario", "decisions", "avg_reward",
                                  "avg_raw_reward", "avg_loss", "epsilon_end",
                                  "wall_clock_sec", "mean_delay"]}
    for a in PHASE_NAMES:
        row[f"phase_{a}_count"] = stats["phase_counts"][a]

    file_exists = os.path.exists(log_path)
    need_header = not file_exists

    if file_exists:
        # 检查已有文件的表头是否和当前字段一致(旧版本代码可能少了 mean_delay 列,
        # 导致后续数据列全部错位)。不一致就自动备份旧文件,重新开一个干净的。
        with open(log_path, encoding="utf-8-sig") as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != fieldnames:
            backup_path = f"{log_path}.bak_{int(time.time())}"
            os.rename(log_path, backup_path)
            print(f"[SUMO Pipeline] 检测到 {log_path} 表头与当前字段不一致(旧版本日志格式),"
                  f"已自动备份为 {backup_path},并重新创建新的日志文件。")
            need_header = True

    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    resolve_scenario_paths(args)
    train_mode = not args.eval
    multi = len(args.scenarios) > 1
    print(f"[SUMO Pipeline] 场景目录={args.sumo_root}")
    print(f"[SUMO Pipeline] 场景数={len(args.scenarios)}: "
          + ", ".join(os.path.splitext(os.path.basename(c))[0] for c, _ in args.scenarios))

    # ── 探测所有场景的状态维度(多场景训练要求完全一致,才能共享同一个网络)
    state_dim = None
    for sumocfg, net in args.scenarios:
        start_sumo(sumocfg, args.seed_start, gui=False)
        probe_builder = SumoStateBuilder(traci, net, tls_id=args.tls_id)
        dim = probe_builder.state_dim
        traci.close()
        if state_dim is None:
            state_dim = dim
        elif dim != state_dim:
            raise ValueError(
                f"多场景训练要求状态维度一致: {sumocfg} 为 {dim} 维, 此前场景为 {state_dim} 维")

    agent = TrafficDQNAgent(
        state_dim=state_dim,
        action_dim=len(PHASE_NAMES),
        epsilon=0.0 if args.eval else 1.0,
        min_replay_size=args.min_replay_size,
        epsilon_decay=args.epsilon_decay,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        target_tau=args.target_update_tau,
        batch_size=args.batch_size,
        n_step=args.n_step,
        lr_warmup_steps=args.lr_warmup,
        lr_total_steps=args.lr_total_steps,
        lr_min_scale=args.lr_min_scale,
        use_huber=(args.loss == "huber"),
    )
    if os.path.exists(args.model) and not args.model.lower().endswith(".json"):
        try:
            agent.load(args.model)
            print(f"[SUMO Pipeline] 已加载已有模型 {args.model}")
            if not args.eval:
                saved_eps = load_epsilon(args.model)
                if saved_eps is not None:
                    agent.epsilon = saved_eps
                    print(f"[SUMO Pipeline] 已恢复上次训练的 epsilon={saved_eps:.4f}(继续探索衰减,而非重新从1.0开始)")
                else:
                    print(f"[SUMO Pipeline] 未找到历史 epsilon 记录,本次从 epsilon={agent.epsilon:.4f} 开始")
        except Exception as exc:
            print(f"[SUMO Pipeline] 跳过旧模型 {args.model}: {exc}")

    print(f"[SUMO Pipeline] 开始{'训练' if train_mode else '评估'},共 {args.episodes} 轮,"
          f"每轮最长 {args.max_sim_time:.0f} 仿真秒, state_dim={state_dim}, "
          f"{'多路口轮换' if multi else '单路口'}模式")

    best_path = os.path.splitext(args.model)[0] + "_best.pt"
    best_delay = float("inf")
    best_per_scenario: dict = {}   # 多场景模式: 各路口各自的最低延误(仅用于日志)

    # ── 断点续训: 如果 episode_log 已经有记录,自动接着往下训,
    #    而不是每次都从 episode 1 / 默认 seed 重新开始(避免新旧数据
    #    混进同一个CSV导致episode编号重复、分析时误判训练效果)。
    resume = get_resume_state(args.episode_log) if train_mode else None
    if resume:
        last_episode, last_seed = resume
        start_episode = last_episode + 1
        # 如果用户没有显式传 --seed-start(还是默认值0),就自动接着上次的seed往后走;
        # 如果用户手动指定了--seed-start,则尊重用户的选择。
        seed_start_for_run = (last_seed + 1) if args.seed_start == 0 else args.seed_start
        print(f"[SUMO Pipeline] 检测到 {args.episode_log} 已有训练记录"
              f"(已完成到 episode {last_episode}),本次将接着从 "
              f"episode {start_episode} 开始训练,seed 从 {seed_start_for_run} 开始"
              f"(而不是重新从1/0开始,避免重复训练相同车流场景)")
    else:
        start_episode = 1
        seed_start_for_run = args.seed_start

    # ── reward尺度校准值: 按路口分别持久化。不同路口拥堵程度差异很大
    #    (本数据集里路口7车均6s、路口18接近80s),共用一套tanh尺度会让
    #    轻度路口全程饱和、重度路口全程无区分度。每个路口用自己的
    #    校准值,跨episode慢速收敛到该路口的真实典型水平。
    calib_ema_rate = 0.15
    default_wait_scale = args.reward_wait_scale
    default_delta_scale = args.reward_delta_scale
    calib = load_calib(args.model) if not args.eval else None
    if calib:
        print(f"[SUMO Pipeline] 已恢复reward尺度校准值({len(calib)}个路口): "
              + ", ".join(f"{t}={w:.0f}/{d:.1f}" for t, (w, d) in sorted(calib.items())))
    else:
        calib = {}

    def calib_for(tag):
        w, d = calib.get(tag, calib.get("_default", (default_wait_scale, default_delta_scale)))
        return max(default_wait_scale, w), max(default_delta_scale, d)

    # ── 多场景轮转顺序: 用本次run的起始seed打乱一次,之后按序循环,
    #    保证每个路口被均匀覆盖(训练=见过尽量多样的路况,评估=逐个测评)。
    order = list(range(len(args.scenarios)))
    random.Random(seed_start_for_run).shuffle(order)

    for offset in range(args.episodes):
        ep = start_episode + offset
        seed = seed_start_for_run + offset
        sumocfg, net = args.scenarios[order[offset % len(order)]]
        tag = os.path.splitext(os.path.basename(sumocfg))[0]
        print(f"\n===== Episode {ep} (本次第{offset+1}/{args.episodes}轮, 场景={tag}, seed={seed}) =====")
        calib_w, calib_d = calib_for(tag)
        stats = run_episode(args, agent, SumoStateBuilder, SumoTrafficEnv,
                            episode_idx=ep, seed=seed, train_mode=train_mode,
                            sumocfg=sumocfg, net=net,
                            calib_wait_scale=calib_w, calib_delta_scale=calib_d)

        print(f"[ep{ep} 汇总][{tag}] 决策步数={stats['decisions']} "
              f"车均延误={stats['mean_delay']:.2f}s "
              f"平均reward={stats['avg_reward']:+.3f} "
              f"平均loss={stats['avg_loss']:.4f} "
              f"epsilon结束={stats['epsilon_end']:.4f} "
              f"lr结束={agent.current_lr:.2e} "
              f"耗时={stats['wall_clock_sec']:.1f}s")
        print(f"  相位分布: {stats['phase_counts']}")
        print(f"  reward尺度校准: wait_scale={calib_w:.1f}->"
              f"(本轮观测{stats['observed_wait_scale']:.1f}) "
              f"delta_scale={calib_d:.1f}->(本轮观测{stats['observed_delta_scale']:.1f})")

        append_episode_log(args.episode_log, stats)

        if train_mode:
            agent.save(args.model)
            save_epsilon(args.model, agent.epsilon)
            # 用该路口本轮观测到的水平,以慢速率校准"它自己"的尺度参考值,
            # 并始终不低于命令行传入的下限(--reward-wait-scale等)。
            w, d = calib_for(tag)
            calib[tag] = (
                max(default_wait_scale,
                    (1 - calib_ema_rate) * w + calib_ema_rate * stats["observed_wait_scale"]),
                max(default_delta_scale,
                    (1 - calib_ema_rate) * d + calib_ema_rate * stats["observed_delta_scale"]),
            )
            save_calib(args.model, calib)
            # 单场景模式保留"_best另存"; 多场景下不同路口的延误不可直接比较,
            # 改为按路口各自记录最低延误(仅日志展示)。
            if multi:
                prev = best_per_scenario.get(tag)
                if stats["mean_delay"] == stats["mean_delay"] and (prev is None or stats["mean_delay"] < prev):
                    best_per_scenario[tag] = stats["mean_delay"]
                    print(f"[SUMO Pipeline] {tag} 训练以来新低延误: {stats['mean_delay']:.2f}s")
            elif stats["mean_delay"] == stats["mean_delay"] and stats["mean_delay"] < best_delay:
                best_delay = stats["mean_delay"]
                agent.save(best_path)
                print(f"[SUMO Pipeline] 本轮延误最低,已另存 {best_path}")

    if train_mode:
        print(f"\n[SUMO Pipeline] 本次 {args.episodes} 轮训练完成"
              f"(累计训练到 episode {start_episode + args.episodes - 1}),模型已保存到 {args.model}")
    else:
        print(f"\n[SUMO Pipeline] 全部 {args.episodes} 轮评估完成")
    print(f"[SUMO Pipeline] 每轮汇总含车均延误 mean_delay,已写入 {args.episode_log}")
    if train_mode and multi and best_per_scenario:
        print("[SUMO Pipeline] 各路口训练期间最低延误: "
              + ", ".join(f"{t}={d:.2f}s" for t, d in sorted(best_per_scenario.items())))
    if train_mode and not multi and best_delay < float("inf"):
        print(f"[SUMO Pipeline] 训练期间最低延误={best_delay:.2f}s,对应 {best_path}")


if __name__ == "__main__":
    main()