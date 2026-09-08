# -*- coding: utf-8 -*-
"""
功能二 · 数据采集模块 —— "端-边-云"三层架构的数据底座
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
端(感知层) : 挂接SUMO逐秒仿真步, 原始采样车辆轨迹(位置/速度/瞬时油耗)、
             车道排队、信号相位状态 —— 统一时间基准, 一切数据可回放对齐。
边(聚合层) : 本地增量聚合, 不等仿真结束就能出指标 ——
             · 进口道排队: 受控车道按"几何进口方向(东南西北)"归并排队长度
             · 累计KPI:   通过量/平均车速/平均等待 O(1)增量更新
             · 停车事件:  在线检测(连续低速≥阈值秒记一次停车), 输出人均停车次数
             · 燃油消耗:  TraCI逐车瞬时油耗(ml/s)积分成总量, 覆盖赛题能耗维度
云(决策层) : collected_data/ 目录即接口 —— dashboard_server.py 实时读取驱
             动ECharts动态仪表盘; visualize_results.py 离线生成时空轨迹图
             与标准化评估报告。云端重启/后接入均可完整回放。

用法(独立采集, 固定配时 vs DQN 各跑一次、同种子可直接对比):
    python data_collector.py --sumocfg intersection_1.sumocfg --control fixed --label fixed
    python data_collector.py --sumocfg intersection_1.sumocfg --control dqn \
        --model dqn_generalist.pt --label dqn
输出: collected_data/<label>/{metrics,queues,phases,trajectories}.csv + run_meta.json
"""
import argparse
import csv
import json
import os
import sys
import time

import sumolib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sumo_pipeline import start_sumo  # noqa: E402  (复用SUMO启动逻辑)
from sumo_traffic_env import SumoTrafficEnv  # noqa: E402
from sumo_state_builder import SumoStateBuilder  # noqa: E402

STOP_SPEED = 0.5          # m/s, 低于此视为停车中
STOP_MIN_SEC = 5          # 连续停车达到该时长才记一次"停车事件"(过滤换道蠕行)
HARD_BRAKE_DECEL = 5.0    # m/s², 减速度超过该值记一次急刹事件(安全指标)。
                          # SUMO车辆常规减速上限4.5, 超过5.0意味着进入非常规制动区间
BRAKE_COOLDOWN_SEC = 3    # 同一车辆急刹事件最小间隔(避免一次制动过程重复计数)
DIRS = ("东", "南", "西", "北")
# TraCI getFuelConsumption 的返回值文档写的是ml/s, 实测是mg/s:
# 巡航(13.89m/s)返回~667, 除以燃油密度745(mg/ml)得0.9ml/s, 折合6.5L/100km,
# 与HBEFA轿车排放模型的标准值吻合 —— 用该系数标定为ml/s
FUEL_UNIT_TO_MLS = 1.0 / 745.0


def approach_direction(net, node, lane_id):
    """按几何位置判断车道所属进口方向: 车道所在边起点节点在路口的哪一侧。
    不依赖边的命名(兼容E_/W_/link_等任意命名), 干线路网也能正确归并。"""
    lane = net.getLane(lane_id)
    edge = lane.getEdge()
    frm = edge.getFromNode()
    if frm.getID().startswith(":") or frm.getID() == node.getID():
        return None
    fx, fy = frm.getCoord()
    x, y = node.getCoord()
    dx, dy = fx - x, fy - y          # 起点节点相对路口的方位
    if abs(dx) >= abs(dy):
        return "西" if dx < 0 else "东"
    return "南" if dy < 0 else "北"


class SimulationDataCollector:
    """挂在仿真步进回调上的逐秒采集器(端层采样 + 边层聚合)。"""

    def __init__(self, traci_conn, net_path, out_dir, label, control_mode,
                 scenario="", seed=0, sample_every=1, traj_every=1):
        self.traci = traci_conn
        self.label = label
        self.control_mode = control_mode
        self.scenario = scenario
        self.seed = seed
        self.sample_every = max(1, int(sample_every))
        self.traj_every = max(1, int(traj_every))

        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        net = sumolib.net.readNet(net_path)
        self._net = net
        self.tls_ids = [n.getID() for n in net.getNodes() if n.getType() == "traffic_light"]

        # ── 边层聚合结构: 车道→(路口,进口方向) 归并表, 建一次用全程 ──
        self.lane_map = {}
        for node in net.getNodes():
            if node.getType() != "traffic_light":
                continue
            for lane_id in traci_conn.trafficlight.getControlledLanes(node.getID()):
                if lane_id not in self.lane_map:
                    d = approach_direction(net, node, lane_id)
                    if d:
                        self.lane_map[lane_id] = (node.getID(), d)
        self.all_lanes = [l for l in traci_conn.lane.getIDList() if not l.startswith(":")]

        # ── 相位语义标注: link索引→进口方向, 用于把每个相位的state串
        #    翻译成人能读懂的标签("东西向绿"/"黄灯过渡"/"全红清空"...),
        #    前端热力图据此按语义着色并配图例 ──
        self.link_dirs = {}
        self._phase_states = {}
        for tls in self.tls_ids:
            dirs = {}
            links = traci_conn.trafficlight.getControlledLinks(tls)
            for idx, conns in enumerate(links or []):
                if not conns:
                    continue
                first = conns[0]
                # SUMO版本差异: 老版本返回dict{inLane:...}, 1.27返回元组(inLane,outLane,via)
                inlane = first.get("inLane") if isinstance(first, dict) else first[0]
                d = self.lane_map.get(inlane, (None, None))[1]
                if d:
                    dirs[idx] = d
            self.link_dirs[tls] = dirs

        self._open_files()
        self._next_sample = 0.0
        self._arrived_cum = 0
        self._inserted_cum = 0
        self._fuel_ml_cum = 0.0
        self._speed_sum = 0.0     # 采样值累加, 用于全程均值
        self._speed_n = 0
        self._wait_sum = 0.0
        self._queue_sum = 0.0
        self._veh_speed = {}      # vid -> 上次采样速度, 用于停车事件在线检测
        self._veh_lowsec = {}     # vid -> 已连续低速秒数
        self.stop_events = 0
        self.stop_seconds = 0
        self._veh_stops = {}      # vid -> 停车次数, 用于二次停车率(安全替代指标)
        self.brake_events = 0     # 急刹事件总数(减速度>4.5m/s², 安全指标)
        self._veh_brake_cool = {} # vid -> 距下次可再记急刹的剩余秒数
        self._finished_veh = 0

    # ────────────────── 文件与元数据 ──────────────────
    def _open_files(self):
        p = lambda name: os.path.join(self.out_dir, name)
        self._f_metrics = open(p("metrics.csv"), "w", newline="", encoding="utf-8")
        self._w_metrics = csv.writer(self._f_metrics)
        self._w_metrics.writerow(["time", "running", "arrived_cum", "inserted_cum",
                                  "mean_speed_mps", "mean_wait_s", "halting",
                                  "fuel_ml_cum", "stop_events", "brake_events", "max_queue"])
        self._f_queues = open(p("queues.csv"), "w", newline="", encoding="utf-8")
        self._w_queues = csv.writer(self._f_queues)
        self._w_queues.writerow(["time", "tls_id", "direction", "queue_len", "veh_count"])
        self._f_phases = open(p("phases.csv"), "w", newline="", encoding="utf-8")
        self._w_phases = csv.writer(self._f_phases)
        self._w_phases.writerow(["time", "tls_id", "phase_index", "state", "green_links"])
        self._f_traj = open(p("trajectories.csv"), "w", newline="", encoding="utf-8")
        self._w_traj = csv.writer(self._f_traj)
        self._w_traj.writerow(["time", "veh_id", "x", "y", "speed", "fuel_mls"])

    def _write_live_state(self, t, row, queues, phases):
        """滚动live_state.json: 仪表盘轮询的轻量当前态(断线重连也不丢全程, 全程在CSV里)"""
        state = {
            "label": self.label, "mode": self.control_mode, "scenario": self.scenario,
            "time": t, "running": row[1], "arrived_cum": row[2],
            "mean_speed_mps": row[4], "mean_wait_s": row[5],
            "fuel_ml_cum": row[7], "stop_events": row[8], "brake_events": row[9],
            "queues": {tls: {d: queues.get((tls, d), 0) for d in DIRS}
                       for tls in self.tls_ids},
            "phases": {tls: {"phase": ph, "state": st} for tls, ph, st in phases},
        }
        tmp = os.path.join(self.out_dir, "live_state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.out_dir, "live_state.json"))

    def _phase_label(self, tls, state):
        """把相位的灯态串翻译成语义标签(需要link方向表)。
        无法判定方向时退化为 绿灯/黄灯/全红 三级。"""
        greens, yellows = set(), False
        for i, ch in enumerate(state):
            if ch in "Gg":
                d = self.link_dirs.get(tls, {}).get(i)
                if d:
                    greens.add(d)
            elif ch == "y":
                yellows = True
        if greens:
            ew, ns = bool(greens & {"东", "西"}), bool(greens & {"南", "北"})
            if ew and ns:
                return "东西+南北绿"
            return "东西向绿" if ew else "南北向绿"
        return "黄灯过渡" if yellows else "全红清空"

    def close(self):
        # 收尾: 把仍在停车的车辆记完
        for vid, low in list(self._veh_lowsec.items()):
            if low >= STOP_MIN_SEC:
                self.stop_events += 1
                self.stop_seconds += low
                self._veh_stops[vid] = self._veh_stops.get(vid, 0) + 1
        self._veh_lowsec.clear()
        for f in (self._f_metrics, self._f_queues, self._f_phases, self._f_traj):
            f.close()
        second_stop_n = sum(1 for v in self._veh_stops.values() if v >= 2)
        meta = {
            "label": self.label, "control_mode": self.control_mode,
            "scenario": self.scenario, "seed": self.seed,
            "sample_every": self.sample_every,
            "tls_ids": self.tls_ids,
            "kpi": {
                "arrived": self._arrived_cum,
                "inserted": self._inserted_cum,
                "mean_speed_mps": round(self._speed_sum / max(self._speed_n, 1), 3),
                "mean_wait_s": round(self._wait_sum / max(self._speed_n, 1), 3),
                "avg_total_queue": round(self._queue_sum / max(self._speed_n, 1), 2),
                "fuel_liters": round(self._fuel_ml_cum / 1000.0, 3),
                "stop_events": self.stop_events,
                "stop_seconds_total": self.stop_seconds,
                "stop_events_per_veh": round(self.stop_events / max(self._finished_veh, 1), 3),
                "harsh_brake_events": self.brake_events,
                "harsh_brake_per_veh": round(self.brake_events / max(self._finished_veh, 1), 3),
                "second_stop_rate": round(second_stop_n / max(self._finished_veh, 1), 3),
            },
        }
        with open(os.path.join(self.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # 相位语义表: {tls: {相位索引: 语义标签}}, 供前端按语义着色+图例
        tls_meta = {}
        for tls in self.tls_ids:
            labels = {str(ph): self._phase_label(tls, st)
                      for (t2, ph), st in self._phase_states.items() if t2 == tls}
            tls_meta[tls] = {"phase_labels": labels}
        with open(os.path.join(self.out_dir, "tls_meta.json"), "w", encoding="utf-8") as f:
            json.dump(tls_meta, f, ensure_ascii=False, indent=2)

    # ────────────────── 核心回调: 每个仿真步调用 ──────────────────
    def on_sim_step(self):
        """仿真每前进一步调用一次。轻量计数每步都累加(不丢车),
        重量级采样(逐车轨迹/逐车道状态)按 sample_every 节流 —— 边缘算力友好。"""
        self._arrived_cum += self.traci.simulation.getArrivedNumber()
        self._inserted_cum += self.traci.simulation.getDepartedNumber()

        t = self.traci.simulation.getTime()
        if t + 1e-6 < self._next_sample:
            return
        self._next_sample = max(self._next_sample + self.sample_every, t + self.sample_every)

        vid_list = self.traci.vehicle.getIDList()

        # ── 网络级KPI: 全车道加权平均车速 + 总排队 ──
        vcount_sum, speed_vsum, halting_sum = 0, 0.0, 0
        for lane_id in self.all_lanes:
            vc = self.traci.lane.getLastStepVehicleNumber(lane_id)
            if vc:
                speed_vsum += vc * self.traci.lane.getLastStepMeanSpeed(lane_id)
            halting_sum += self.traci.lane.getLastStepHaltingNumber(lane_id)
            vcount_sum += vc
        mean_speed = speed_vsum / vcount_sum if vcount_sum else 0.0

        # ── 逐车: 等待/轨迹/油耗 + 停车事件在线检测 ──
        interval = self.sample_every
        wait_sum, fuel_ml = 0.0, 0.0
        do_traj = (round(t) % self.traj_every == 0)
        cur_ids = set()
        for vid in vid_list:
            cur_ids.add(vid)
            speed = self.traci.vehicle.getSpeed(vid)
            wait_sum += self.traci.vehicle.getWaitingTime(vid)
            fuel = self.traci.vehicle.getFuelConsumption(vid) * FUEL_UNIT_TO_MLS
            fuel_ml += fuel * interval
            if do_traj:
                x, y = self.traci.vehicle.getPosition(vid)
                self._w_traj.writerow([f"{t:.0f}", vid, f"{x:.1f}", f"{y:.1f}",
                                       f"{speed:.2f}", f"{fuel:.3f}"])
            # 急刹事件: 采样点瞬时减速度超阈值记一次, 冷却期内不重复计数
            acc = self.traci.vehicle.getAcceleration(vid)
            cool = self._veh_brake_cool.get(vid, 0.0) - interval
            if acc <= -HARD_BRAKE_DECEL and cool <= 0:
                self.brake_events += 1
                cool = BRAKE_COOLDOWN_SEC
            self._veh_brake_cool[vid] = max(cool, 0.0)
            # 停车事件: 连续低速计时, 恢复行驶或离网时结算
            if speed < STOP_SPEED:
                self._veh_lowsec[vid] = self._veh_lowsec.get(vid, 0) + interval
            else:
                low = self._veh_lowsec.pop(vid, 0)
                if low >= STOP_MIN_SEC:
                    self.stop_events += 1
                    self.stop_seconds += low
                    self._veh_stops[vid] = self._veh_stops.get(vid, 0) + 1
            self._veh_speed[vid] = speed
        for vid in list(self._veh_lowsec.keys() - cur_ids):
            low = self._veh_lowsec.pop(vid, 0)
            if low >= STOP_MIN_SEC:
                self.stop_events += 1
                self.stop_seconds += low
                self._veh_stops[vid] = self._veh_stops.get(vid, 0) + 1
            self._veh_speed.pop(vid, None)
            self._veh_brake_cool.pop(vid, None)
        self._finished_veh = self._arrived_cum
        self._fuel_ml_cum += fuel_ml

        # ── 进口道排队聚合 + 相位状态 ──
        queues = {}
        queue_veh = {}
        for lane_id, (tls_id, d) in self.lane_map.items():
            key = (tls_id, d)
            queues[key] = queues.get(key, 0) + self.traci.lane.getLastStepHaltingNumber(lane_id)
            queue_veh[key] = queue_veh.get(key, 0) + self.traci.lane.getLastStepVehicleNumber(lane_id)
        for (tls_id, d), q in sorted(queues.items()):
            self._w_queues.writerow([f"{t:.0f}", tls_id, d, q, queue_veh[(tls_id, d)]])
        max_queue = max(queues.values()) if queues else 0

        phases = []
        for tls_id in self.tls_ids:
            ph = self.traci.trafficlight.getPhase(tls_id)
            state = self.traci.trafficlight.getRedYellowGreenState(tls_id)
            phases.append((tls_id, ph, state))
            self._phase_states.setdefault((tls_id, ph), state)
            self._w_phases.writerow([f"{t:.0f}", tls_id, ph, state, state.count("G")])

        mean_wait = wait_sum / len(vid_list) if vid_list else 0.0
        row = (f"{t:.0f}", len(vid_list), self._arrived_cum, self._inserted_cum,
               f"{mean_speed:.3f}", f"{mean_wait:.2f}", halting_sum,
               f"{self._fuel_ml_cum:.0f}", self.stop_events, self.brake_events, max_queue)
        self._w_metrics.writerow(row)

        self._speed_sum += mean_speed
        self._speed_n += 1
        self._wait_sum += mean_wait
        self._queue_sum += sum(queues.values())

        self._f_metrics.flush()
        self._f_queues.flush()
        self._f_phases.flush()
        if do_traj:
            self._f_traj.flush()
        self._write_live_state(t, row, queues, phases)


def run_collection(args):
    """独立采集入口: 同一场景同一种子下分别采集固定配时与DQN, 数据天然可比。
    --program 可切换基线信号方案(如干线畸形coord程序切回net内置的'0')。"""
    from sumo_pipeline import parse_net_from_sumocfg
    import traci

    sumocfg = args.sumocfg if os.path.isfile(args.sumocfg) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sumo", args.sumocfg)
    sumocfg = os.path.abspath(sumocfg)
    net = parse_net_from_sumocfg(sumocfg)
    out_dir = os.path.join(args.out_dir, args.label)

    start_sumo(sumocfg, args.seed, gui=args.gui)
    _net_obj = sumolib.net.readNet(net)
    tls_ids = [n.getID() for n in _net_obj.getNodes() if n.getType() == "traffic_light"]
    if args.program:
        for t in tls_ids:
            traci.trafficlight.setProgram(t, args.program)
        print(f"[采集] 已切换信号程序: {args.program}")

    collector = SimulationDataCollector(
        traci, net, out_dir, args.label, args.control,
        scenario=os.path.basename(sumocfg), seed=args.seed,
        sample_every=args.sample_every, traj_every=args.traj_every)
    print(f"[采集] 开始: label={args.label} 模式={args.control} 最长{args.seconds}s "
          f"输出={out_dir}")

    t0 = time.time()
    try:
        if args.control == "dqn":
            from traffic_dqn import TrafficDQNAgent
            builder = SumoStateBuilder(traci, net, tls_id=args.tls_id)
            env = SumoTrafficEnv(
                traci, net, tls_id=args.tls_id,
                min_green_sec=args.min_green, max_green_sec=args.max_green,
                max_sim_time=args.seconds,
                step_callback=collector.on_sim_step)
            env.reset()
            agent = TrafficDQNAgent(state_dim=builder.state_dim, action_dim=4, epsilon=0.0)
            agent.load(args.model)
            print(f"[采集] DQN模式: 已加载 {args.model} (state_dim={builder.state_dim}, "
                  f"tls={env.tls_ids})")

            def build_all_states():
                tls_phases = {t: (env.current_phase[t], env.phase_duration_sec[t])
                              for t in env.tls_ids}
                return {t: builder.build(t, env.current_phase[t],
                                         env.phase_duration_sec[t], tls_phases)
                        for t in env.tls_ids}

            current_states = build_all_states()
            decisions = 0
            while not env.is_finished():
                actions = {}
                for t in env.tls_ids:
                    a = agent.select_action(current_states[t], greedy=True)
                    actions[t] = env.clamp_action(t, a)
                env.apply_actions(actions)
                current_states = build_all_states()
                decisions += 1
            print(f"[采集] DQN共{decisions}次决策 × {len(env.tls_ids)}个路口")
        else:
            # 固定配时: 不干预信号, 原始方案自行运行, 采集器只旁观记录
            while (traci.simulation.getTime() < args.seconds
                   and traci.simulation.getMinExpectedNumber() > 0):
                traci.simulationStep()
                collector.on_sim_step()
    finally:
        collector.close()
        traci.close()

    with open(os.path.join(out_dir, "run_meta.json"), encoding="utf-8") as f:
        kpi = json.load(f)["kpi"]
    print(f"[采集] 完成, 耗时{time.time()-t0:.1f}s. 核心KPI: {json.dumps(kpi, ensure_ascii=False)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="功能二·数据采集模块(端-边-云架构的端+边层)")
    p.add_argument("--sumocfg", required=True, help="sumocfg路径或sumo/目录下的文件名")
    p.add_argument("--control", default="fixed", choices=["fixed", "dqn"],
                   help="fixed=原始固定配时(不干预信号); dqn=加载模型贪心控制")
    p.add_argument("--model", default="dqn_generalist.pt", help="dqn模式的模型路径")
    p.add_argument("--label", default=None, help="本次采集标签(输出子目录名), 默认=控制模式")
    p.add_argument("--out-dir", default="collected_data", help="采集数据根目录")
    p.add_argument("--seconds", default=900, type=float, help="采集时长(仿真秒)")
    p.add_argument("--seed", default=0, type=int, help="SUMO随机种子(对比实验必须一致)")
    p.add_argument("--min-green", default=10, type=int)
    p.add_argument("--max-green", default=60, type=int)
    p.add_argument("--sample-every", default=1, type=int, help="KPI采样间隔(仿真秒)")
    p.add_argument("--traj-every", default=1, type=int, help="轨迹采样间隔(仿真秒)")
    p.add_argument("--tls-id", default=None, help="信号灯ID, 默认None=自动接管路网内全部信号灯")
    p.add_argument("--program", default=None,
                   help="基线信号程序programID(如'0'=net内置方案)。干线场景默认加载的"
                        "coord程序状态串畸形(走廊方向永无绿灯), 对比实验请用--program 0")
    p.add_argument("--gui", action="store_true", help="用sumo-gui可视化运行(演示录屏用)")
    args = p.parse_args()
    if args.label is None:
        args.label = args.control
    run_collection(args)


if __name__ == "__main__":
    main()
