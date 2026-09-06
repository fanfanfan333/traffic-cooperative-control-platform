# -*- coding: utf-8 -*-
"""
训练曲线绘图工具 —— 把 sumo_pipeline.py 写出的 episode_log CSV 画成图。

用法:
    python plot_training.py dqn_generalist.csv                  # 单文件, 输出同名.png
    python plot_training.py dqn_i1_v2.csv -o curves_i1.png      # 指定输出
    python plot_training.py a.csv b.csv --compare               # 多文件同图对比(延误)

单路口日志画: 延误曲线 + loss曲线(对数轴)。
多路口日志(scenario列): 延误按路口着色散点 + 全局滚动均值,
便于检查"是不是所有路口都在变好"而不仅是平均值。
"""
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")           # 无显示环境也能出图
import matplotlib.pyplot as plt

# Windows 自带中文字体,避免图例/轴标签变方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_log(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {"ep": [], "delay": [], "loss": [], "scenario": []}
    for r in rows:
        try:
            out["ep"].append(int(r["episode"]))
            out["delay"].append(float(r["mean_delay"]))
        except (KeyError, ValueError, TypeError):
            continue
        loss = r.get("avg_loss", "")
        try:
            out["loss"].append(float(loss))
        except (ValueError, TypeError):
            out["loss"].append(float("nan"))
        out["scenario"].append(r.get("scenario", ""))
    return out


def rolling(xs, window=10):
    n = len(xs)
    out = []
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = [v for v in xs[lo:i + 1] if v == v]   # 过滤NaN
        out.append(sum(seg) / len(seg) if seg else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description="episode_log 训练曲线绘图")
    p.add_argument("logs", nargs="+", help="episode_log CSV 文件(可多个)")
    p.add_argument("-o", "--output", default=None, help="输出PNG路径(默认取第一个文件名)")
    p.add_argument("--window", default=10, type=int, help="滚动均值窗口(默认10轮)")
    p.add_argument("--compare", action="store_true",
                   help="多文件画到同一张延误图上对比(忽略scenario着色)")
    args = p.parse_args()

    data = [read_log(path) for path in args.logs]
    if not data or not any(d["ep"] for d in data):
        print("没有可画的episode记录")
        return

    out_path = args.output or (
        args.logs[0].rsplit(".", 1)[0] + ".png" if len(args.logs) == 1
        else "training_compare.png")

    if args.compare and len(data) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for path, d in zip(args.logs, data):
            label = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            axes[0].plot(d["ep"], d["delay"], alpha=0.25)
            axes[0].plot(d["ep"], rolling(d["delay"], args.window), label=label)
            axes[1].plot(d["ep"], rolling(d["loss"], args.window), label=label)
        axes[0].set_title("车均延误 (粗线=滚动均值)")
        axes[1].set_title("avg_loss (滚动均值, 对数轴)")
        axes[1].set_yscale("log")
    else:
        d = data[0]
        multi = any(s for s in d["scenario"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        if multi:
            scenes = sorted({s for s in d["scenario"] if s})
            cmap = plt.cm.tab20
            for i, s in enumerate(scenes):
                xs = [ep for ep, sc in zip(d["ep"], d["scenario"]) if sc == s]
                ys = [v for ep, sc, v in zip(d["ep"], d["scenario"], d["delay"]) if sc == s]
                axes[0].scatter(xs, ys, s=14, color=cmap(i % 20), label=s.replace("intersection_", "路口"))
            axes[0].legend(fontsize=6, ncol=2)
        else:
            axes[0].plot(d["ep"], d["delay"], alpha=0.35, label="每轮")
        axes[0].plot(d["ep"], rolling(d["delay"], args.window), "k-", lw=2, label="滚动均值")
        axes[0].set_title("车均延误 (黑线=滚动均值)")
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("车均延误 (s)")

        losses = [v for v in d["loss"] if v == v]
        if losses:
            axes[1].plot(d["ep"], d["loss"], alpha=0.3)
            axes[1].plot(d["ep"], rolling(d["loss"], args.window), "k-", lw=2)
            axes[1].set_yscale("log")
            axes[1].set_title("avg_loss (对数轴)")
            axes[1].set_xlabel("episode")
        else:
            axes[1].text(0.5, 0.5, "无loss数据", ha="center", va="center")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
