# -*- coding: utf-8 -*-
"""
比赛口径延误打分: 解析 tripinfo 车均 waitingTime。
也可对静态绿灯做 Webster 初值 + 爬山(不依赖 TraCI/DQN)。
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def mean_waiting_from_tripinfo(path: str) -> float:
    if not path or not os.path.isfile(path):
        return float("nan")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return float("nan")
    trips = root.findall("tripinfo")
    if not trips:
        return float("nan")
    total = 0.0
    for trip in trips:
        total += float(trip.get("waitingTime", 0.0))
    return total / len(trips)


def read_green_durations(net_path: str, tls_id: str = "J"):
    tree = ET.parse(net_path)
    tl = None
    for node in tree.findall(".//tlLogic"):
        if tls_id is None or node.get("id") == tls_id:
            tl = node
            break
    if tl is None:
        tl = tree.find(".//tlLogic")
    if tl is None:
        return [], tree
    greens = []
    for phase in tl.findall("phase"):
        state = phase.get("state", "")
        if "G" in state or "g" in state:
            greens.append(int(float(phase.get("duration", "10"))))
    return greens, tree


def write_green_durations(tree: ET.ElementTree, greens, tls_id: str = "J", path: str = None):
    tl = None
    for node in tree.findall(".//tlLogic"):
        if tls_id is None or node.get("id") == tls_id:
            tl = node
            break
    if tl is None:
        tl = tree.find(".//tlLogic")
    gi = 0
    for phase in tl.findall("phase"):
        state = phase.get("state", "")
        if "G" in state or "g" in state:
            if gi < len(greens):
                phase.set("duration", str(int(greens[gi])))
                gi += 1
    if path:
        tree.write(path, encoding="utf-8", xml_declaration=True)


def webster_greens(orig_greens, sat=1800 * 2, lost=4, y_ratio=0.55):
    """没有流量表时,用原始绿灯比例 + Webster 周期公式给一组可行初值。"""
    n = max(len(orig_greens), 1)
    y = min(max(y_ratio, 0.2), 0.85)
    cycle = max(60, min(180, int((1.5 * n * lost + 5) / (1 - y))))
    eff = max(cycle - n * lost, n * 10)
    weights = orig_greens if orig_greens and sum(orig_greens) > 0 else [1] * n
    total_w = float(sum(weights))
    greens = [max(10, int(w / total_w * eff)) for w in weights]
    drift = eff - sum(greens)
    if greens:
        greens[0] = max(10, greens[0] + drift)
    return greens


def run_fixed_sumo(sumo_bin: str, sumocfg: str, net_path: str, seed: int, tripinfo_path: str) -> float:
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg))
    os.makedirs(os.path.dirname(tripinfo_path), exist_ok=True)
    cmd = [
        sumo_bin, "-c", os.path.abspath(sumocfg),
        "--net-file", os.path.abspath(net_path),
        "--no-warnings", "--seed", str(seed),
        "--tripinfo-output", tripinfo_path,
        "--duration-log.statistics",
    ]
    subprocess.run(cmd, cwd=cfg_dir, check=False, capture_output=True, timeout=180)
    return mean_waiting_from_tripinfo(tripinfo_path)


def hill_climb(sumo_bin, sumocfg, net_path, greens, seed, work_dir, tls_id="J", rounds=20):
    greens, tree = list(greens), ET.parse(net_path)
    best = list(greens)
    tmp_net = os.path.join(work_dir, "cand.net.xml")
    write_green_durations(tree, best, tls_id, tmp_net)
    best_d = run_fixed_sumo(sumo_bin, sumocfg, tmp_net, seed, os.path.join(work_dir, "trip_best.xml"))
    for _ in range(rounds):
        improved = False
        for pos in range(len(best)):
            for step in (+5, -5):
                cand = list(best)
                cand[pos] = max(10, cand[pos] + step)
                write_green_durations(ET.parse(net_path), cand, tls_id, tmp_net)
                delay = run_fixed_sumo(
                    sumo_bin, sumocfg, tmp_net, seed, os.path.join(work_dir, "trip_cand.xml")
                )
                if delay == delay and delay < best_d:
                    best, best_d = cand, delay
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_d


def get_sumo_bin(sumo_home: str) -> str:
    exe = os.path.join(sumo_home, "bin", "sumo.exe")
    return exe if os.path.isfile(exe) else "sumo"


def main():
    p = argparse.ArgumentParser(description="静态配时延误打分 / Webster / 爬山")
    p.add_argument("--sumocfg", required=True)
    p.add_argument("--net", default=None)
    p.add_argument("--tls-id", default="J")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--hill-climb", action="store_true")
    p.add_argument("--sumo-home", default=os.environ.get("SUMO_HOME", r"D:\Sumo"))
    args = p.parse_args()
    cfg = os.path.abspath(args.sumocfg)
    net = os.path.abspath(args.net) if args.net else os.path.join(
        os.path.dirname(cfg), "net", os.path.splitext(os.path.basename(cfg))[0] + ".net.xml"
    )
    sumo_bin = get_sumo_bin(args.sumo_home)
    orig, _ = read_green_durations(net, args.tls_id)
    web = webster_greens(orig)
    work = tempfile.mkdtemp(prefix="sumo_score_")
    try:
        orig_d = run_fixed_sumo(sumo_bin, cfg, net, args.seed, os.path.join(work, "orig.xml"))
        tmp_web = os.path.join(work, "web.net.xml")
        shutil.copy2(net, tmp_web)
        write_green_durations(ET.parse(tmp_web), web, args.tls_id, tmp_web)
        web_d = run_fixed_sumo(sumo_bin, cfg, tmp_web, args.seed, os.path.join(work, "web.xml"))
        print(f"原始配时 greens={orig} 延误={orig_d:.2f}s")
        print(f"Webster   greens={web} 延误={web_d:.2f}s")
        if args.hill_climb:
            best, best_d = hill_climb(sumo_bin, cfg, net, web, args.seed, work, args.tls_id)
            print(f"爬山最优 greens={best} 延误={best_d:.2f}s")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
