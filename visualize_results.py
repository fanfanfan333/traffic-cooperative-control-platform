# -*- coding: utf-8 -*-
"""
功能二 · 离线评估报告生成器(云端分析层)

输入: data_collector.py / sumo_pipeline.py --collect-data 产出的采集目录
输出: reports/ 下的图表PNG + 标准化评估报告Markdown
    1. 时空轨迹图          —— 赛题答题要求明确点名(横轴=空间, 纵轴=时间,
                              颜色=车速, 自动叠加检测出的"停车区段"红点)
    2. 排队长度对比曲线     —— 四个进口方向, 两方案叠加
    3. 平均车速/累计通过对比 —— 指标变化曲线
    4. 核心KPI汇总图        —— 含相对改善百分比标注
    5. report.md           —— 标准化格式报告(效率/能耗/舒适性多维指标表+结论),
                                服务赛题"为地方标准提供参考"的要求

用法:
    python visualize_results.py --baseline fixed --proposal dqn
    python visualize_results.py --baseline fixed --proposal dqn --seconds-note "600s仿真,seed=0"
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIRS = ("东", "南", "西", "北")
STOP_SPEED = 0.5
STOP_MIN_SEC = 5


def load_run(data_dir, label):
    run_dir = os.path.join(data_dir, label)
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到 {meta_path}, 请先用 data_collector.py 采集(标签={label})")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    def rows(name):
        with open(os.path.join(run_dir, name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    return {"label": label, "meta": meta, "dir": run_dir,
            "metrics": rows("metrics.csv"), "queues": rows("queues.csv"),
            "traj": rows("trajectories.csv")}


def detect_stops(traj_rows):
    """从轨迹里检测停车事件: 同一车连续低速≥STOP_MIN_SEC记一次。
    返回 (停车区段列表[(veh, t0, t1, x均值)], 每车停车次数均值)。"""
    events, cur = [], {}
    by_veh = {}
    for r in traj_rows:
        by_veh.setdefault(r["veh_id"], []).append(r)
    for veh, rs in by_veh.items():
        rs.sort(key=lambda r: float(r["time"]))
        low_start, low_x = None, 0.0
        count = 0
        for r in rs + [None]:
            slow = r is not None and float(r["speed"]) < STOP_SPEED
            if slow:
                if low_start is None:
                    low_start, low_x = float(r["time"]), [float(r["x"])]
                else:
                    low_x.append(float(r["x"]))
            else:
                if low_start is not None:
                    t1 = float(rs[-1]["time"]) if r is None else float(r["time"])
                    if t1 - low_start >= STOP_MIN_SEC:
                        events.append((veh, low_start, t1, sum(low_x) / len(low_x)))
                        count += 1
                low_start = None
        cur[veh] = count
    return events, (sum(cur.values()) / max(len(cur), 1))


def plot_timespace(run, events, out_path):
    traj = run["traj"]
    if not traj:
        print(f"[报告] {run['label']} 无轨迹数据, 跳过时空图")
        return
    xs = [float(r["x"]) for r in traj]
    ts = [float(r["time"]) for r in traj]
    sp = [float(r["speed"]) for r in traj]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(xs, ts, c=sp, s=1.5, alpha=0.5, cmap="RdYlGn", vmin=0, vmax=13.9)
    # 叠加自动检测的停车区段(红点), 让"车堵在哪、堵多久"一眼可见
    if events:
        ex = [e[3] for e in events]
        et0 = [e[1] for e in events]
        ax.scatter(ex, et0, s=14, c="red", marker="x", alpha=0.8,
                   label=f"停车事件(自动检测, n={len(events)})")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("沿走廊位置 x (m)")
    ax.set_ylabel("仿真时间 (s)")
    ax.set_title(f"时空轨迹图 — {run['label']} ({run['meta']['control_mode']})\n"
                 f"颜色=车速(红停绿行), 红叉=自动检测的停车事件")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="车速 (m/s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_queues(runs, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = ["-", "--"]
    for k, run in enumerate(runs):
        by_time = {}
        for r in run["queues"]:
            by_time.setdefault(int(r["time"]), {})
            by_time[int(r["time"])][r["direction"]] = \
                by_time[int(r["time"])].get(r["direction"], 0) + int(r["queue_len"])
        times = sorted(by_time)
        for i, d in enumerate(DIRS):
            vals = [by_time[t].get(d, 0) for t in times]
            ax.plot(times, vals, styles[k], alpha=0.85, color=f"C{i}",
                    label=f"{run['label']}-{d}" if k == 0 or i == 0 else None)
    ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("排队长度 (冻结车辆数)")
    ax.set_title("各进口道排队长度对比 (实线=" + runs[0]["label"] + ", 虚线=" + runs[-1]["label"] + ")")
    ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_speed_arrived(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for run, style in zip(runs, ["-", "--"]):
        t = [int(r["time"]) for r in run["metrics"]]
        axes[0].plot(t, [float(r["mean_speed_mps"]) * 3.6 for r in run["metrics"]],
                     style, label=run["label"])
        axes[1].plot(t, [int(r["arrived_cum"]) for r in run["metrics"]],
                     style, label=run["label"])
    axes[0].set_title("网络平均车速 (km/h)"); axes[1].set_title("累计通过车辆 (辆)")
    for ax in axes:
        ax.set_xlabel("仿真时间 (s)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_kpi(kpis, out_path):
    """kpis: [(指标名, 基准值, 方案值, 越低越好?)]"""
    names = [k[0] for k in kpis]
    base = [k[1] for k in kpis]
    prop = [k[2] for k in kpis]
    import numpy as np
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(y + 0.2, base, height=0.4, label="固定配时(基准)", color="#f59e0b")
    ax.barh(y - 0.2, prop, height=0.4, label="DQN协同管控", color="#37d2c5")
    for i, (_, b, p, lower_better) in enumerate(kpis):
        if b:
            pct = (p - b) / b * 100
            good = pct < 0 if lower_better else pct > 0
            ax.text(max(b, p) * 1.02, i, f"{pct:+.1f}%",
                    color="#4ade80" if good else "#f87171", fontsize=10, va="center")
    ax.set_yticks(y, names, fontsize=10)
    ax.set_xlabel("数值")
    ax.set_title("核心KPI对比 (标注=DQN相对基准的变化, 绿=改善)")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="功能二·标准化评估报告生成器(云端分析层)")
    p.add_argument("--baseline", default="fixed", help="基准方案采集标签(默认fixed)")
    p.add_argument("--proposal", default="dqn", help="对比方案采集标签(默认dqn)")
    p.add_argument("--data-dir", default="collected_data")
    p.add_argument("-o", "--out-dir", default="reports")
    p.add_argument("--note", default="", help="报告里的实验条件备注(如仿真时长/种子)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = load_run(args.data_dir, args.baseline)
    prop = load_run(args.data_dir, args.proposal)

    # ── 1. 时空轨迹图(含自动停车检测) ──
    for run in (base, prop):
        events, per_veh = detect_stops(run["traj"])
        run["stop_events_detected"] = events
        run["stops_per_veh"] = per_veh
        plot_timespace(run, events, os.path.join(args.out_dir, f"timespace_{run['label']}.png"))
        print(f"[报告] 时空轨迹图: timespace_{run['label']}.png (停车事件{len(events)}起, "
              f"人均{per_veh:.2f}次)")

    # ── 2/3. 对比曲线 ──
    plot_queues([base, prop], os.path.join(args.out_dir, "queues_compare.png"))
    plot_speed_arrived([base, prop], os.path.join(args.out_dir, "speed_arrived_compare.png"))
    print("[报告] 对比曲线: queues_compare.png / speed_arrived_compare.png")

    # ── 4. KPI汇总 ──
    kb, kp = base["meta"]["kpi"], prop["meta"]["kpi"]
    kpis = [
        ("平均等待 (s/车)", kb["mean_wait_s"], kp["mean_wait_s"], True),
        ("平均车速 (m/s)", kb["mean_speed_mps"], kp["mean_speed_mps"], False),
        ("平均总排队 (辆)", kb["avg_total_queue"], kp["avg_total_queue"], True),
        ("累计通过 (辆)", kb["arrived"], kp["arrived"], False),
        ("总油耗 (L)", kb["fuel_liters"], kp["fuel_liters"], True),
        ("人均停车 (次/车)", kb["stop_events_per_veh"], kp["stop_events_per_veh"], True),
    ]
    # 安全维度(急刹/二次停车), 旧版采集数据无该字段时自动跳过
    safety_rows = []
    if kb.get("harsh_brake_per_veh") is not None and kp.get("harsh_brake_per_veh") is not None:
        safety_rows.append(("车均急刹事件 (次/车)", kb["harsh_brake_per_veh"], kp["harsh_brake_per_veh"], True))
    if kb.get("second_stop_rate") is not None and kp.get("second_stop_rate") is not None:
        safety_rows.append(("二次停车率", kb["second_stop_rate"], kp["second_stop_rate"], True))
    kpis += safety_rows
    plot_kpi(kpis, os.path.join(args.out_dir, "kpi_summary.png"))
    print("[报告] KPI汇总图: kpi_summary.png")

    # ── 5. 标准化Markdown报告 ──
    def pct_line(name, b, p, lower_better, nd=2):
        chg = (p - b) / b * 100 if b else 0
        good = chg < 0 if lower_better else chg > 0
        mark = "✅改善" if good else "⚠️劣化"
        return f"| {name} | {fmt(b, nd)} | {fmt(p, nd)} | {chg:+.1f}% | {mark} |"

    lines = [
        "# 协同管控算法评估报告",
        "",
        f"- 对比方案: **{prop['label']}**(DQN协同管控) vs **{base['label']}**(固定配时基准)",
        f"- 实验条件: 场景={prop['meta']['scenario']}, 随机种子={prop['meta']['seed']}, "
        f"采集时长={prop['metrics'][-1]['time']}s {args.note}",
        f"- 数据来源: data_collector.py 逐秒采集(轨迹/排队/相位/油耗), 同种子同负荷直接可比",
        "",
        "## 核心指标对比",
        "",
        "| 指标 | 固定配时(基准) | DQN协同管控 | 变化 | 判定 |",
        "|------|--------------|------------|------|------|",
        pct_line("平均等待时间 (s/车)", kb["mean_wait_s"], kp["mean_wait_s"], True),
        pct_line("平均车速 (m/s)", kb["mean_speed_mps"], kp["mean_speed_mps"], False),
        pct_line("平均总排队长度 (辆)", kb["avg_total_queue"], kp["avg_total_queue"], True),
        pct_line("累计通过车辆 (辆)", kb["arrived"], kp["arrived"], False),
        pct_line("总燃油消耗 (L)", kb["fuel_liters"], kp["fuel_liters"], True),
        pct_line("人均停车次数 (次/车)", kb["stop_events_per_veh"], kp["stop_events_per_veh"], True),
    ]
    for name, b, p, lb in safety_rows:
        lines.append(pct_line(name, b, p, lb, nd=3))
    lines += [
        "",
        "## 图表索引",
        "",
        f"- `timespace_{base['label']}.png` / `timespace_{prop['label']}.png`: 时空轨迹图"
        "(颜色=车速, 红叉=自动检测的停车事件)",
        "- `queues_compare.png`: 各进口道排队长度对比",
        "- `speed_arrived_compare.png`: 平均车速 / 累计通过对比",
        "- `kpi_summary.png`: 核心KPI汇总",
        "",
        "## 结论",
        "",
        f"DQN协同管控相对固定配时: 平均等待时间变化 "
        f"{(kp['mean_wait_s']-kb['mean_wait_s'])/kb['mean_wait_s']*100:+.1f}%, "
        f"通行能力(累计通过)变化 "
        f"{(kp['arrived']-kb['arrived'])/kb['arrived']*100:+.1f}%, "
        f"油耗变化 {(kp['fuel_liters']-kb['fuel_liters'])/kb['fuel_liters']*100:+.1f}%, "
        f"停车次数变化 {(kp['stop_events_per_veh']-kb['stop_events_per_veh'])/kb['stop_events_per_veh']*100:+.1f}%。",
        ("效率、能耗、舒适性(停车次数)、安全性(急刹/二次停车率)四个维度均纳入评估, "
         "满足标准化测试指标集要求。" if safety_rows else
         "效率、能耗、舒适性(停车次数)三个维度均纳入评估, 满足标准化测试指标集要求。"),
        "",
    ]
    report_path = os.path.join(args.out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] 标准化报告: {report_path}")


if __name__ == "__main__":
    main()
