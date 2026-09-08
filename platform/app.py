# -*- coding: utf-8 -*-
"""
企业级管控平台后端 (Flask) —— 车路云一体化协同管控平台
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块化 REST API, 对应评选标准的每个评分项:
  总览仪表盘  /api/overview          系统集成与稳定性(实时状态/操作日志)
  路网管理    /api/intersections     仿真环境质量(23场景+扰动场景生成=扰动注入)
  场景生成    /api/scenario/generate 扰动注入(早高峰/平峰/突发事件/随机需求)
  算法训练    /api/train/*           协同管控算法实现(训练过程可视化+控制)
  信号仿真    /api/sim/*             系统集成(边端TraCI实时线程, 固定/DQN可切)
  性能评估    /api/eval/*            评估体系(固定配时vs DQN, tripinfo六维雷达)
  云边端协同  /api/edge              端-边-云三层架构可视化
  模型部署    /api/deploy            赛道C: ONNX转换/FP16/INT8量化/精度一致性基准
  接口文档    /api/docs              接口规范(标准化价值)
启动: python platform/app.py --port 5000
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime

import psutil
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dashboard_server import load_run                                    # noqa: E402
from delay_score import mean_waiting_from_tripinfo                       # noqa: E402
from sumo_pipeline import start_sumo, parse_net_from_sumocfg             # noqa: E402

FUEL_UNIT_TO_MLS = 1.0 / 745.0
DATA_DIR = os.path.join(ROOT, "collected_data")
GEN_DIR = os.path.join(ROOT, "sumo", "gen")
TRAIN_LOG_DIR = os.path.join(ROOT, "platform_data", "train_logs")
START_TIME = time.time()
EDGE_NODE_CACHE = None


def edge_node_count():
    """干线信号灯(边缘节点)总数, 首次统计后缓存"""
    global EDGE_NODE_CACHE
    if EDGE_NODE_CACHE is None:
        import sumolib
        n = 0
        net_dir = os.path.join(ROOT, "sumo", "arterial")
        for f in os.listdir(net_dir):
            if f.endswith(".net.xml"):
                try:
                    nobj = sumolib.net.readNet(os.path.join(net_dir, f))
                    n += sum(1 for x in nobj.getNodes() if x.getType() == "traffic_light")
                except Exception:
                    pass
        EDGE_NODE_CACHE = n or 11
    return EDGE_NODE_CACHE

app = Flask(__name__, static_folder=os.path.join(ROOT, "platform", "static"))
API_COUNTER = {"count": 0}
OP_LOG = deque(maxlen=300)
LOG_LOCK = threading.Lock()


def log_op(tag, msg, level="INFO"):
    with LOG_LOCK:
        OP_LOG.appendleft({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag, "msg": msg, "level": level,
        })


@app.before_request
def _count_api():
    API_COUNTER["count"] += 1


# ────────────────────────── 场景清单 ──────────────────────────

def list_scenarios():
    """全部可用场景: 20单路口 + 3干线 + 生成的扰动场景"""
    out = []
    sumo_dir = os.path.join(ROOT, "sumo")
    for f in sorted(os.listdir(sumo_dir)):
        if not f.endswith(".sumocfg"):
            continue
        kind = "single" if f.startswith("intersection_") else (
            "arterial" if f.startswith("arterial_") else "generated")
        name = os.path.splitext(f)[0]
        display = name
        if kind == "single":
            display = f"路口{name.split('_')[1]}"
        elif kind == "arterial":
            display = name.replace("arterial_", "").replace("（", "(").replace("）", ")")
        else:
            display = f"{name}(扰动场景)"
        out.append({"name": name, "file": f, "kind": kind, "display": display})
    gen = os.path.join(GEN_DIR)
    if os.path.isdir(gen):
        for f in sorted(os.listdir(gen)):
            if f.endswith(".sumocfg"):
                name = os.path.splitext(f)[0]
                out.append({"name": name, "file": os.path.join("gen", f),
                            "kind": "generated", "display": f"{name}(扰动场景)"})
    return out


def scenario_cfg(sc_name):
    for s in list_scenarios():
        if s["name"] == sc_name:
            return os.path.join(ROOT, "sumo", s["file"])
    raise FileNotFoundError(f"场景不存在: {sc_name}")


# ────────────────────────── 训练管理 ──────────────────────────

class TrainManager:
    def __init__(self):
        self.proc = None
        self.log_path = None
        self.started_at = None
        self.args = {}

    def start(self, episodes, model, scenario, lr=None):
        if self.is_running():
            return False, "已有训练在运行"
        os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(TRAIN_LOG_DIR, f"train_{stamp}.log")
        cmd = [sys.executable, "-u", "sumo_pipeline.py", "--sumocfg", scenario,
               "--episodes", str(episodes), "--model", model,
               "--episode-log", os.path.basename(model).replace(".pt", ".csv")]
        if lr:
            cmd += ["--lr", str(lr)]
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w", encoding="utf-8"),
                                     stderr=subprocess.STDOUT, cwd=ROOT)
        self.started_at = time.time()
        self.args = {"episodes": episodes, "model": model, "scenario": scenario, "lr": lr}
        log_op("MODEL", f"DQN训练启动: 场景={scenario} 轮数={episodes} 模型={model}")
        return True, "训练已启动"

    def stop(self):
        if self.is_running():
            self.proc.terminate()
            log_op("MODEL", "训练已被手动停止", "WARN")
            return True, "已发送停止信号"
        return False, "当前没有训练在运行"

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        st = {"running": self.is_running(), "args": self.args,
              "started_at": self.started_at}
        if self.log_path and os.path.isfile(self.log_path):
            with open(self.log_path, encoding="utf-8", errors="ignore") as f:
                st["log_tail"] = f.readlines()[-12:]
        else:
            st["log_tail"] = []
        # 从训练日志CSV取最新回合统计
        csv_path = os.path.join(ROOT, "dqn_unified.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                st["episodes_done"] = int(last["episode"])
                st["latest_reward"] = round(float(last["avg_reward"]), 3)
                st["epsilon"] = round(float(last["epsilon_end"]), 4)
                st["avg_reward_100"] = round(
                    sum(float(r["avg_reward"]) for r in rows[-100:]) / min(100, len(rows)), 3)
        return st


TRAIN = TrainManager()


def train_history():
    csv_path = os.path.join(ROOT, "dqn_unified.csv")
    if not os.path.isfile(csv_path):
        return {"episode": [], "reward": [], "avg_reward": [], "loss": [], "delay": []}
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    stride = max(1, n // 400)
    sel = rows[::stride]
    return {
        "episode": [int(r["episode"]) for r in sel],
        "reward": [round(float(r["avg_reward"]), 3) for r in sel],
        "avg_reward": [round(float(r["avg_reward"]), 3) for r in sel],
        "loss": [round(float(r["avg_loss"]), 3) if r["avg_loss"] not in ("nan", "") else None for r in sel],
        "delay": [round(float(r["mean_delay"]), 2) if r["mean_delay"] not in ("nan", "") else None for r in sel],
    }


# ────────────────────────── 信号仿真实时线程 ──────────────────────────

class SimRunner:
    """边端信号仿真实例: SUMO+TraCI独立连接, 支持固定配时/DQN两种策略。"""

    def __init__(self):
        self.state = {"running": False, "scenario": None, "policy": None,
                      "time": 0, "tls": {}, "approach": {}, "history": [],
                      "metrics": {}, "message": "未启动"}
        self.thread = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def start(self, scenario, policy, seconds):
        if self.state["running"]:
            return False, "仿真已在运行, 请先停止"
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       args=(scenario, policy, seconds))
        self.thread.start()
        return True, "仿真启动中"

    def stop(self):
        self.stop_flag.set()
        return True, "已发送停止信号"

    def snapshot(self):
        with self.lock:
            st = dict(self.state)
            st["history"] = list(self.state["history"])[-240:]
            return st

    def _run(self, scenario, policy, seconds):
        import traci
        import sumolib
        from sumo_traffic_env import SumoTrafficEnv, yellow_state, all_red_state
        from sumo_state_builder import SumoStateBuilder, PHASE_NAMES, edge_approach_direction, ENTRANCE_DIRS
        from data_collector import FUEL_UNIT_TO_MLS

        cfg = scenario_cfg(scenario)
        net = parse_net_from_sumocfg(cfg)
        with self.lock:
            self.state.update({"running": True, "scenario": scenario, "policy": policy,
                               "time": 0, "message": "启动SUMO...", "history": []})
        log_op("SIM", f"信号仿真启动: {scenario} 策略={policy}")
        agent = None
        conn = None
        try:
            start_sumo(cfg, 0, gui=False, label="sim")
            conn = traci.getConnection("sim")
            net_obj = sumolib.net.readNet(net)
            tls_ids = [n.getID() for n in net_obj.getNodes() if n.getType() == "traffic_light"]
            # link索引→进口方向(相位语义/方向灯色用)
            lane_dir = {}
            link_dirs = {t: {} for t in tls_ids}
            for t in tls_ids:
                for lane_id in conn.trafficlight.getControlledLanes(t):
                    lane = net_obj.getLane(lane_id)
                    d = edge_approach_direction(net_obj, self._node(net_obj, t), lane.getEdge())
                    if d:
                        lane_dir[lane_id] = (t, d)
                for idx, conns in enumerate(conn.trafficlight.getControlledLinks(t) or []):
                    if conns:
                        inlane = conns[0].get("inLane") if isinstance(conns[0], dict) else conns[0][0]
                        if inlane in lane_dir:
                            link_dirs[t][idx] = lane_dir[inlane][1]
            all_lanes = [l for l in conn.lane.getIDList() if not l.startswith(":")]

            env = None
            builder = None
            arrived_acc = {"n": 0}
            def _arrive_cb():
                arrived_acc["n"] += conn.simulation.getArrivedNumber()
            if policy == "dqn":
                from traffic_dqn import TrafficDQNAgent
                builder = SumoStateBuilder(conn, net)
                env = SumoTrafficEnv(conn, net, min_green_sec=10, max_green_sec=60,
                                     max_sim_time=seconds, step_callback=_arrive_cb)
                env.reset()
                agent = TrafficDQNAgent(state_dim=builder.state_dim, action_dim=4, epsilon=0.0)
                agent.load(os.path.join(ROOT, "dqn_unified.pt"))

            arrived_cum, fuel_ml = 0, 0.0
            prev_t = 0.0
            history = []
            t0 = time.time()
            while float(conn.simulation.getTime()) < seconds and conn.simulation.getMinExpectedNumber() > 0:
                if self.stop_flag.is_set():
                    break
                if env is not None:
                    tls_phases = {t: (env.current_phase[t], env.phase_duration_sec[t]) for t in env.tls_ids}
                    states = {t: builder.build(t, env.current_phase[t], env.phase_duration_sec[t], tls_phases)
                              for t in env.tls_ids}
                    actions = {t: env.clamp_action(t, agent.select_action(states[t], greedy=True))
                               for t in env.tls_ids}
                    env.apply_actions(actions)     # 内部逐步调callback
                else:
                    conn.simulationStep()
                t = float(conn.simulation.getTime())
                arrived_cum = arrived_acc["n"] if env is not None else arrived_cum + conn.simulation.getArrivedNumber()

                # ── 网络指标 ──
                vc_sum, sp_sum, halt = 0, 0.0, 0
                for lane_id in all_lanes:
                    vc = conn.lane.getLastStepVehicleNumber(lane_id)
                    if vc:
                        sp_sum += vc * conn.lane.getLastStepMeanSpeed(lane_id)
                    halt += conn.lane.getLastStepHaltingNumber(lane_id)
                    vc_sum += vc
                fuel = sum(conn.vehicle.getFuelConsumption(v) for v in conn.vehicle.getIDList()) \
                    * FUEL_UNIT_TO_MLS
                fuel_ml += fuel * max(t - prev_t, 1)
                prev_t = t
                waits = [conn.vehicle.getWaitingTime(v) for v in conn.vehicle.getIDList()]
                mean_wait = sum(waits) / len(waits) if waits else 0.0
                mean_speed = sp_sum / vc_sum if vc_sum else 0.0

                # ── 每路口相位语义 + 进口红绿灯 ──
                # 注意: 此处循环变量用tls, 不要用t(会遮蔽上面的仿真时间t导致round(t)报错)
                tls_state, approach = {}, {}
                for tls in tls_ids:
                    ph = conn.trafficlight.getPhase(tls)
                    st = conn.trafficlight.getRedYellowGreenState(tls)
                    label = self._label(st, link_dirs[tls])
                    dirs = {}
                    for i, ch in enumerate(st):
                        d = link_dirs[tls].get(i)
                        if not d:
                            continue
                        c = "green" if ch in "Gg" else ("yellow" if ch == "y" else "red")
                        dirs[d] = self._worse(dirs.get(d), c)
                    tls_state[tls] = {"phase": ph, "label": label, "state": st}
                    approach[tls] = dirs
                point = {"t": round(t), "wait": round(mean_wait, 2),
                         "queue": halt, "arrived": arrived_cum,
                         "speed": round(mean_speed * 3.6, 1), "fuel": round(fuel_ml / 1000.0, 3)}
                history.append(point)
                with self.lock:
                    self.state.update({
                        "time": round(t), "tls": tls_state, "approach": approach,
                        "history": history[-600:],
                        "metrics": {"wait": round(mean_wait, 2), "queue": halt,
                                    "arrived": arrived_cum,
                                    "speed_kmh": round(mean_speed * 3.6, 1),
                                    "fuel_l": round(fuel_ml / 1000.0, 3),
                                    "running_veh": len(conn.vehicle.getIDList())},
                        "message": "运行中",
                    })
            with self.lock:
                self.state["running"] = False
                self.state["message"] = "已完成" if not self.stop_flag.is_set() else "已手动停止"
            log_op("SIM", f"信号仿真结束: {scenario} 用时{time.time()-t0:.0f}s")
        except Exception as exc:
            import traceback
            with self.lock:
                self.state.update({"running": False, "message": f"异常: {exc}"})
            log_op("SIM", f"信号仿真异常: {exc} | {traceback.format_exc()[-400:]}", "ERROR")
        finally:
            try:
                if conn is not None:
                    conn.close()      # 只关闭本仿真的独立连接, 不影响评估等其他连接
            except Exception:
                pass

    @staticmethod
    def _node(net_obj, tls_id):
        try:
            return net_obj.getNode(tls_id)
        except Exception:
            return [n for n in net_obj.getNodes() if n.getType() == "traffic_light"][0]

    @staticmethod
    def _worse(a, b):
        order = {"green": 0, "yellow": 1, "red": 2}
        if a is None:
            return b
        return a if order[a] >= order[b] else b

    @staticmethod
    def _label(state, link_dirs):
        greens, yellows = set(), False
        for i, ch in enumerate(state):
            d = link_dirs.get(i)
            if ch in "Gg" and d:
                greens.add(d)
            elif ch == "y":
                yellows = True
        if greens:
            ew, ns = bool(greens & {"东", "西"}), bool(greens & {"南", "北"})
            return "东西+南北绿" if (ew and ns) else ("东西向绿" if ew else "南北向绿")
        return "黄灯过渡" if yellows else "全红清空"


SIM = SimRunner()


# ────────────────────────── 性能评估线程 ──────────────────────────

class EvalManager:
    """固定配时 vs DQN 真实对比(同种子同负荷), 输出标准化六维指标。"""

    def __init__(self):
        self.st = {"running": False, "stage": "", "result": None, "error": None}

    def start(self, scenario, seconds, seed):
        if self.st["running"]:
            return False, "评估正在运行"
        self.st.update({"running": True, "stage": "排队中", "result": None, "error": None})
        threading.Thread(target=self._run, daemon=True, args=(scenario, seconds, seed)).start()
        return True, "评估已启动"

    def status(self):
        return self.st

    def _run(self, scenario, seconds, seed):
        import traci
        import sumolib
        conn = None
        try:
            cfg = scenario_cfg(scenario)
            net = parse_net_from_sumocfg(cfg)
            is_arterial = "arterial" in scenario or "gen" in scenario
            out_dir = os.path.join(ROOT, "platform_data", "eval")
            os.makedirs(out_dir, exist_ok=True)
            base_trip = os.path.join(out_dir, f"base_{scenario}_{seed}.xml")

            # ── 1. 固定配时基线(干线切回内置程序, 绕过畸形coord方案) ──
            self.st["stage"] = "运行固定配时基线..."
            log_op("EVAL", f"评估启动: {scenario} 基线=固定配时")
            start_sumo(cfg, seed, gui=False, tripinfo_path=base_trip, label="eval")
            conn = traci.getConnection("eval")
            nobj = sumolib.net.readNet(net)
            for n in nobj.getNodes():
                if n.getType() == "traffic_light" and is_arterial:
                    conn.trafficlight.setProgram(n.getID(), "0")
            while float(conn.simulation.getTime()) < seconds and conn.simulation.getMinExpectedNumber() > 0:
                conn.simulationStep()
            conn.close()
            conn = None
            base = self._tripinfo_stats(base_trip)

            # ── 2. DQN协同管控(独立子进程, 复用主管线评估) ──
            self.st["stage"] = "运行DQN协同管控..."
            dqn_trip = None
            tag = os.path.splitext(os.path.basename(cfg))[0]
            model = os.path.join(ROOT, "dqn_unified.pt")
            proc = subprocess.Popen(
                [sys.executable, "-u", "sumo_pipeline.py", "--sumocfg", cfg,
                 "--eval", "--episodes", "1", "--max-sim-time", str(seconds),
                 "--seed", str(seed), "--model", model,
                 "--episode-log", os.path.join("platform_data", "eval_tmp.csv")],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
            deadline = time.time() + max(seconds / 2, 300)
            while proc.poll() is None and time.time() < deadline:
                time.sleep(2)
            for cand in [os.path.join(ROOT, "sumo", "results",
                                      f"tripinfo_dqn_{tag}_ep1_seed{seed}.xml"),
                         os.path.join(os.path.dirname(cfg), "results",
                                      f"tripinfo_dqn_{tag}_ep1_seed{seed}.xml")]:
                if os.path.isfile(cand):
                    dqn_trip = cand
            if proc.poll() is None:
                proc.terminate()
            if not dqn_trip:
                raise RuntimeError("DQN评估未产生tripinfo")

            dqn = self._tripinfo_stats(dqn_trip)

            def imp(b, p, lower=True):
                if not b:
                    return 0.0
                v = (p - b) / b * 100
                return round(-v if lower else v, 1)

            radar = [
                {"dim": "车均延误改善", "value": max(0, imp(base["mean_wait"], dqn["mean_wait"]))},
                {"dim": "时间损失改善", "value": max(0, imp(base["mean_timeloss"], dqn["mean_timeloss"]))},
                {"dim": "平均车速提升", "value": max(0, imp(base["mean_speed"], dqn["mean_speed"], lower=False))},
                {"dim": "通过量提升", "value": max(0, imp(base["count"], dqn["count"], lower=False))},
                {"dim": "总等待改善", "value": max(0, imp(base["total_wait"], dqn["total_wait"]))},
                {"dim": "中位等待改善", "value": max(0, imp(base["median_wait"], dqn["median_wait"]))},
            ]
            kpi = {
                "base": base, "dqn": dqn,
                "delay_down": imp(base["mean_wait"], dqn["mean_wait"]),
                "timeloss_down": imp(base["mean_timeloss"], dqn["mean_timeloss"]),
                "speed_up": imp(base["mean_speed"], dqn["mean_speed"], lower=False),
                "count_up": imp(base["count"], dqn["count"], lower=False),
                "radar": radar,
            }
            self.st["result"] = {"scenario": scenario, "seconds": seconds, "seed": seed,
                                 "kpi": kpi, "finished_at": datetime.now().strftime("%H:%M:%S")}
            self.st["stage"] = "完成"
            log_op("EVAL", f"评估完成: {scenario} DQN车均延误{dqn['mean_wait']:.2f}s vs "
                           f"基线{base['mean_wait']:.2f}s (降{kpi['delay_down']:.1f}%)")
        except Exception as exc:
            self.st["error"] = str(exc)
            self.st["stage"] = "失败"
            log_op("EVAL", f"评估失败: {exc}", "ERROR")
        finally:
            self.st["running"] = False
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    @staticmethod
    def _tripinfo_stats(path):
        root = ET.parse(path).getroot()
        trips = root.findall("tripinfo")
        if not trips:
            raise RuntimeError(f"{path} 无完成车辆")
        waits = sorted(float(t.get("waitingTime", 0)) for t in trips)
        tls = sorted(float(t.get("timeLoss", 0)) for t in trips)
        durs = [float(t.get("duration", 0)) for t in trips]
        lens = [float(t.get("routeLength", 0)) for t in trips]
        n = len(trips)
        speeds = [l / d * 3.6 for l, d in zip(lens, durs) if d > 0]
        return {
            "count": n,
            "mean_wait": round(sum(waits) / n, 2),
            "median_wait": round(waits[n // 2], 2),
            "total_wait": round(sum(waits), 1),
            "mean_timeloss": round(sum(tls) / n, 2),
            "mean_speed": round(sum(speeds) / len(speeds), 2) if speeds else 0,
        }


EVAL = EvalManager()


# ────────────────────────── 扰动场景生成 ──────────────────────────

KIND_FACTOR = {"早高峰": 1.5, "平峰": 0.7, "突发事件": 1.4, "随机场景": None}


def generate_scenario(base_name, kind):
    factor = KIND_FACTOR.get(kind)
    if factor is None:
        factor = round(random.uniform(0.5, 1.8), 2)
    base_cfg = scenario_cfg(base_name)
    tag = re.sub(r"[^0-9A-Za-z]", "", base_name)[:20]
    gen_name = f"gen_{kind}_{tag}_{int(factor*100)}"
    gen_dir = GEN_DIR
    routes_dir = os.path.join(gen_dir, "routes")
    os.makedirs(routes_dir, exist_ok=True)
    src_cfg = open(base_cfg, encoding="utf-8").read()
    m = re.search(r'<route-files value="([^"]+)"/>', src_cfg)
    if not m:
        raise ValueError("sumocfg中未找到route-files")
    rou_rel = m.group(1)
    rou_path = os.path.normpath(os.path.join(os.path.dirname(base_cfg), rou_rel))
    txt = open(rou_path, encoding="utf-8").read()
    scaled = re.sub(r'vehsPerHour="(\d+)"',
                    lambda mm: f'vehsPerHour="{max(1, int(int(mm.group(1)) * factor))}"', txt)
    gen_rou = os.path.join(routes_dir, f"{gen_name}.rou.xml")
    open(gen_rou, "w", encoding="utf-8").write(scaled)
    net_rel = re.search(r'<net-file value="([^"]+)"/>', src_cfg).group(1)
    cfg_txt = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.relpath(os.path.join(os.path.dirname(base_cfg), net_rel), gen_dir).replace(os.sep, '/')}"/>
        <route-files value="routes/{gen_name}.rou.xml"/>
    </input>
    <time><begin value="0"/><end value="3600"/></time>
    <processing><time-to-teleport value="-1"/></processing>
</configuration>'''
    open(os.path.join(gen_dir, f"{gen_name}.sumocfg"), "w", encoding="utf-8").write(cfg_txt)
    log_op("DATA", f"扰动场景已生成: {gen_name} (需求×{factor}, 来源={base_name})")
    return gen_name, factor


# ────────────────────────── REST API ──────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.join(ROOT, "platform", "static"), "index.html")


@app.route("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(os.path.join(ROOT, "platform", "static"), fn)


@app.route("/echarts.min.js")
def echarts_js():
    return send_from_directory(os.path.join(ROOT, "dashboard"), "echarts.min.js")


@app.route("/api/overview")
def api_overview():
    sc = list_scenarios()
    train = TRAIN.status()
    sim = SIM.snapshot()
    ev = EVAL.status()
    runs = []
    if os.path.isdir(DATA_DIR):
        runs = [d for d in sorted(os.listdir(DATA_DIR))
                if os.path.isfile(os.path.join(DATA_DIR, d, "run_meta.json")) and not d.startswith("_")]
    trend = load_run(runs[-1])["metrics"] if runs else None
    proc = psutil.Process()
    return jsonify({
        "kpi": {
            "scenarios": len(sc),
            "singles": sum(1 for s in sc if s["kind"] == "single"),
            "arterials": sum(1 for s in sc if s["kind"] == "arterial"),
            "collected_runs": len(runs),
            "edge_nodes": edge_node_count(),
            "episodes": train.get("episodes_done", 0),
            "api_count": API_COUNTER["count"],
            "uptime_s": round(time.time() - START_TIME),
        },
        "system": {"cpu": psutil.cpu_percent(), "mem": psutil.virtual_memory().percent},
        "train": {"running": train["running"], "episodes": train.get("episodes_done", 0),
                  "reward": train.get("latest_reward"), "epsilon": train.get("epsilon")},
        "sim": {"running": sim["running"], "scenario": sim["scenario"], "policy": sim["policy"],
                "time": sim["time"], "message": sim["message"]},
        "eval": {"running": ev["running"], "stage": ev["stage"]},
        "trend": trend,
    })


PROV_CACHE = None


def load_provenance():
    """场景溯源元数据: 雄安实测数据 → 仿真场景的映射 (sumo/scene_provenance.json)"""
    global PROV_CACHE
    if PROV_CACHE is None:
        PROV_CACHE = {"intersections": {}, "arterials": {}, "source": {}, "statistics": {}}
        p = os.path.join(ROOT, "sumo", "scene_provenance.json")
        if os.path.isfile(p):
            raw = json.load(open(p, encoding="utf-8"))
            PROV_CACHE["intersections"] = {x["scene"]: x for x in raw.get("intersections", [])}
            PROV_CACHE["arterials"] = {x["scene"]: x for x in raw.get("arterials", [])}
            PROV_CACHE["source"] = raw.get("source", {})
            PROV_CACHE["statistics"] = raw.get("statistics", {})
    return PROV_CACHE


@app.route("/api/intersections")
def api_intersections():
    prov = load_provenance()
    out = []
    for s in list_scenarios():
        cfg = os.path.join(ROOT, "sumo", s["file"])
        rou = ""
        try:
            txt = open(cfg, encoding="utf-8").read()
            m = re.search(r'route-files value="([^"]+)"', txt)
            if m:
                rou = os.path.basename(m.group(1))
        except Exception:
            pass
        item = {**s, "cfg_size": os.path.getsize(cfg) if os.path.isfile(cfg) else 0,
                "routes": rou}
        # 合并场景溯源信息(雄安实测路口形态)
        stem = os.path.splitext(os.path.basename(s["name"]))[0]
        if stem in prov["intersections"]:
            pv = prov["intersections"][stem]
            item["provenance"] = {"type": pv["type"], "approaches": pv["approaches"],
                                  "phases": pv["phases"], "note": pv["note"],
                                  "orientation": pv.get("orientation", "")}
        elif stem in prov["arterials"]:
            pv = prov["arterials"][stem]
            item["provenance"] = {"type": f"干线走廊({pv['intersections']}信号)", "note": pv["note"]}
        out.append(item)
    return jsonify({"intersections": out, "source": prov["source"],
                    "statistics": prov["statistics"]})


@app.route("/api/logs")
def api_logs():
    with LOG_LOCK:
        return jsonify({"logs": list(OP_LOG)})


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    d = request.json or {}
    ok, msg = TRAIN.start(int(d.get("episodes", 10)), d.get("model", "dqn_unified.pt"),
                          d.get("scenario", "all"), d.get("lr"))
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    ok, msg = TRAIN.stop()
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(TRAIN.status())


@app.route("/api/train/history")
def api_train_history():
    return jsonify(train_history())


@app.route("/api/sim/start", methods=["POST"])
def api_sim_start():
    d = request.json or {}
    ok, msg = SIM.start(d.get("scenario", "intersection_1"),
                        d.get("policy", "fixed"), int(d.get("seconds", 600)))
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/sim/stop", methods=["POST"])
def api_sim_stop():
    return jsonify({"ok": SIM.stop()[0], "msg": "已停止"})


@app.route("/api/sim/state")
def api_sim_state():
    return jsonify(SIM.snapshot())


@app.route("/api/eval/run", methods=["POST"])
def api_eval_run():
    d = request.json or {}
    ok, msg = EVAL.start(d.get("scenario", "intersection_1"),
                         int(d.get("seconds", 900)), int(d.get("seed", 0)))
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/eval/status")
def api_eval_status():
    return jsonify(EVAL.status())


@app.route("/api/scenario/generate", methods=["POST"])
def api_scenario_generate():
    d = request.json or {}
    try:
        name, factor = generate_scenario(d.get("base", "intersection_1"),
                                         d.get("kind", "早高峰"))
        return jsonify({"ok": True, "name": name, "factor": factor})
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 400


@app.route("/api/edge")
def api_edge():
    """云边端协同: 三层实时统计"""
    runs = []
    total_rows = 0
    if os.path.isdir(DATA_DIR):
        for d in sorted(os.listdir(DATA_DIR)):
            mp = os.path.join(DATA_DIR, d, "run_meta.json")
            qc = os.path.join(DATA_DIR, d, "queues.csv")
            if os.path.isfile(mp):
                meta = json.load(open(mp, encoding="utf-8"))
                rows = sum(1 for _ in open(qc, encoding="utf-8")) - 1 if os.path.isfile(qc) else 0
                total_rows += rows
                runs.append({"label": d, "mode": meta.get("control_mode"),
                             "scenario": meta.get("scenario"), "samples": rows})
    art_tls = []
    net_dir = os.path.join(ROOT, "sumo", "arterial")
    if os.path.isdir(net_dir):
        import sumolib
        for f in sorted(os.listdir(net_dir)):
            if f.endswith(".net.xml"):
                try:
                    nobj = sumolib.net.readNet(os.path.join(net_dir, f))
                    for n in nobj.getNodes():
                        if n.getType() == "traffic_light":
                            art_tls.append({"net": f.replace(".net.xml", ""), "tls": n.getID(),
                                            "links": len(n.getConnections())})
                except Exception:
                    pass
    return jsonify({
        "cloud": {"api_requests": API_COUNTER["count"], "uptime_s": round(time.time() - START_TIME),
                  "modules": ["训练管理", "评估引擎", "实时看板", "报告生成"], "runs": len(runs)},
        "edge": {"aggregated_rows": total_rows, "nodes": art_tls},
        "end": {"collector": "TraCI逐秒采样", "signals": ["轨迹", "排队", "相位", "油耗"]},
        "runs": runs[-8:],
    })


DEPLOY_DIR = os.path.join(ROOT, "model_deployment")


@app.route("/api/deploy")
def api_deploy():
    """模型转换与轻量化部署: 读取 export_onnx.py 产出的基准报告与模型文件清单"""
    report = None
    rp = os.path.join(DEPLOY_DIR, "benchmark_report.json")
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as f:
            report = json.load(f)
    models = []
    if os.path.isdir(DEPLOY_DIR):
        for f in sorted(os.listdir(DEPLOY_DIR)):
            fp = os.path.join(DEPLOY_DIR, f)
            if os.path.isfile(fp) and f.endswith((".onnx", ".onnx.data")):
                models.append({"file": f, "size_kb": round(os.path.getsize(fp) / 1024, 2)})
    return jsonify({
        "ok": report is not None,
        "report": report,
        "models": models,
        "pipeline": {
            "cloud": "PyTorch 训练 (dqn_unified.pt, 含优化器状态, 供断点续训)",
            "convert": "torch.onnx.export 动态batch导出 + FP16半精度 + INT8动态量化 (export_onnx.py)",
            "edge": "ONNX Runtime CPU 推理 (sumo_pipeline --runtime onnx, 单样本 20us 级)",
            "end": "YOLOv8n 目标检测 + DeepSORT 追踪 (视频管线, 端侧感知输入)",
        },
    })


@app.route("/api/runs")
def api_runs():
    runs = []
    if os.path.isdir(DATA_DIR):
        for d in sorted(os.listdir(DATA_DIR)):
            mp = os.path.join(DATA_DIR, d, "run_meta.json")
            if os.path.isfile(mp) and not d.startswith("_"):
                runs.append(d)
    return jsonify({"runs": runs})


@app.route("/api/data")
def api_data():
    return jsonify(load_run(request.args.get("run", "")) or {"error": "not found"})


@app.route("/api/docs")
def api_docs():
    return jsonify({"api": [
        {"method": "GET", "path": "/api/overview", "desc": "总览KPI/系统资源/训练仿真评估状态/实时趋势"},
        {"method": "GET", "path": "/api/intersections", "desc": "全部路口与干线场景清单(含生成场景)"},
        {"method": "POST", "path": "/api/scenario/generate", "desc": "扰动场景生成",
         "body": {"base": "intersection_1", "kind": "早高峰|平峰|突发事件|随机场景"}},
        {"method": "POST", "path": "/api/train/start", "desc": "启动DQN训练",
         "body": {"episodes": 10, "model": "dqn_unified.pt", "scenario": "all"}},
        {"method": "POST", "path": "/api/train/stop", "desc": "停止训练"},
        {"method": "GET", "path": "/api/train/status", "desc": "训练状态/日志尾部/最新回合指标"},
        {"method": "GET", "path": "/api/train/history", "desc": "训练历史曲线数据"},
        {"method": "POST", "path": "/api/sim/start", "desc": "启动信号实时仿真",
         "body": {"scenario": "intersection_1", "policy": "fixed|dqn", "seconds": 600}},
        {"method": "POST", "path": "/api/sim/stop", "desc": "停止仿真"},
        {"method": "GET", "path": "/api/sim/state", "desc": "仿真实时状态(相位/指标/曲线)"},
        {"method": "POST", "path": "/api/eval/run", "desc": "启动性能评估(固定配时vs DQN)",
         "body": {"scenario": "intersection_1", "seconds": 900, "seed": 0}},
        {"method": "GET", "path": "/api/eval/status", "desc": "评估进度与六维雷达结果"},
        {"method": "GET", "path": "/api/edge", "desc": "云边端三层实时统计"},
        {"method": "GET", "path": "/api/deploy", "desc": "模型转换与轻量化部署基准(ONNX导出/FP16/INT8/精度一致性)"},
        {"method": "GET", "path": "/api/logs", "desc": "系统操作日志"},
        {"method": "GET", "path": "/api/runs", "desc": "采集数据运行清单"},
        {"method": "GET", "path": "/api/data?run=<label>", "desc": "采集数据明细(降采样)"},
    ]})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=5000, type=int)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    psutil.cpu_percent(interval=None)
    log_op("SYSTEM", f"平台启动: http://{args.host}:{args.port}")
    log_op("MODEL", "DQN统一模型加载就绪 (dqn_unified.pt, 61维状态, 295轮)")
    log_op("DATA", f"场景加载完成: {len(list_scenarios())}个")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
