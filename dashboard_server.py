# -*- coding: utf-8 -*-
"""
功能二 · 云端服务 —— 轻量实时仪表盘服务器(仅用Python标准库, 零额外依赖)。

数据流: data_collector.py / sumo_pipeline.py --collect-data 把逐秒数据写入
collected_data/<label>/*.csv, 本服务器按需读取并向浏览器提供JSON接口,
ECharts前端每2秒轮询刷新 —— 采集与展示完全解耦, 仿真结束后仍可回放。

用法:
    python dashboard_server.py                # 默认 http://127.0.0.1:8080
    python dashboard_server.py --port 8090    # 换端口
然后浏览器打开 http://127.0.0.1:8080 即可(演示录屏建议F11全屏)。
"""
import argparse
import csv
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "collected_data")
MAX_POINTS = 1200            # 每条曲线最多返回的点数(自动降采样)


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def downsample_indices(n):
    stride = max(1, (n + MAX_POINTS - 1) // MAX_POINTS)
    return list(range(0, n, stride)), stride


def load_run(label):
    run_dir = os.path.join(DATA_DIR, label)
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    metrics_rows = read_csv(os.path.join(run_dir, "metrics.csv"))
    idx, stride = downsample_indices(len(metrics_rows))
    pick = lambda row, key, cast=float: (cast(row[key]) if row.get(key) not in (None, "", "nan") else None)
    metrics = {
        "time": [int(metrics_rows[i]["time"]) for i in idx],
        "running": [int(metrics_rows[i]["running"]) for i in idx],
        "arrived_cum": [int(metrics_rows[i]["arrived_cum"]) for i in idx],
        "speed_kmh": [round(pick(metrics_rows[i], "mean_speed_mps") * 3.6, 2) if metrics_rows[i].get("mean_speed_mps") else None for i in idx],
        "mean_wait_s": [pick(metrics_rows[i], "mean_wait_s") for i in idx],
        "halting": [int(metrics_rows[i]["halting"]) for i in idx],
        "fuel_ml_cum": [pick(metrics_rows[i], "fuel_ml_cum") for i in idx],
        "stop_events": [int(metrics_rows[i]["stop_events"]) for i in idx],
    }

    # 排队: 每时刻按方向求和(跨所有信号灯) → {方向: [q, ...]}
    q_rows = read_csv(os.path.join(run_dir, "queues.csv"))
    q_by_time = {}
    for r in q_rows:
        t = int(r["time"])
        q_by_time.setdefault(t, {})
        q_by_time[t][r["direction"]] = q_by_time[t].get(r["direction"], 0) + int(r["queue_len"])
    times_q = sorted(q_by_time)
    qidx, _ = downsample_indices(len(times_q))
    queues = {}
    for d in ("东", "南", "西", "北"):
        queues[d] = [q_by_time[times_q[i]].get(d, 0) for i in qidx]

    # 相位语义: 每个路口每个相位 → 语义标签 → 统一语义色码
    #   0=东西向绿 1=南北向绿 2=东西+南北绿 3=绿灯(方向未知) 4=黄灯过渡 5=全红清空
    SEM = {
        "东西向绿": (0, "#34d399"), "南北向绿": (1, "#38bdf8"),
        "东西+南北绿": (2, "#a3e635"), "绿灯": (3, "#4ade80"),
        "黄灯过渡": (4, "#fbbf24"), "全红清空": (5, "#64748b"),
    }
    tls_meta = {}
    meta_path2 = os.path.join(run_dir, "tls_meta.json")
    if os.path.isfile(meta_path2):
        with open(meta_path2, encoding="utf-8") as f:
            tls_meta = json.load(f)

    # 相位: 矩阵[tls序][时间序] = 语义色码(热力图用), 外加图例
    p_rows = read_csv(os.path.join(run_dir, "phases.csv"))
    tls_list, times_p = [], []
    p_state = {}   # (tls, phase) -> state串
    p_map = {}     # (time, tls) -> phase索引
    for r in p_rows:
        t, tls, ph = int(r["time"]), r["tls_id"], int(r["phase_index"])
        if tls not in tls_list:
            tls_list.append(tls)
        if not times_p or times_p[-1] != t:
            times_p.append(t)
        p_state[(tls, ph)] = r.get("state", "")
        p_map[(t, tls)] = ph

    def phase_semantic(tls, ph):
        meta_tls = tls_meta.get(tls, {}).get("phase_labels", {})
        if str(ph) in meta_tls:
            return SEM.get(meta_tls[str(ph)], SEM["绿灯"])[0]
        state = p_state.get((tls, ph), "")
        if "y" in state and "G" not in state and "g" not in state:
            return SEM["黄灯过渡"][0]
        if "G" in state or "g" in state:
            return SEM["绿灯"][0]
        return SEM["全红清空"][0]

    pidx, pstride = downsample_indices(len(times_p))
    matrix = [[phase_semantic(tls, p_map.get((times_p[i], tls), 0)) for i in pidx]
              for tls in tls_list]
    used = sorted({c for row in matrix for c in row})
    legend = [{"code": c,
               "label": next(k for k, v in SEM.items() if v[0] == c),
               "color": next(v[1] for v in SEM.values() if v[0] == c)}
              for c in used]

    return {
        "label": label, "meta": meta, "metrics": metrics, "queues": queues,
        "phases": {"tls": tls_list, "matrix": matrix, "legend": legend,
                   "stride": pstride * (metrics["time"][1] - metrics["time"][0] if len(metrics["time"]) > 1 else 1)},
        "sig": f"{len(metrics_rows)}_{metrics_rows[-1]['time'] if metrics_rows else 0}",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                with open(os.path.join(ROOT, "dashboard", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/echarts.min.js":
                with open(os.path.join(ROOT, "dashboard", "echarts.min.js"), "rb") as f:
                    return self._send(200, f.read(), "application/javascript")
            if url.path == "/api/runs":
                runs = []
                if os.path.isdir(DATA_DIR):
                    for d in sorted(os.listdir(DATA_DIR)):
                        mp = os.path.join(DATA_DIR, d, "run_meta.json")
                        if d.startswith("_") or not os.path.isfile(mp):
                            continue
                        try:
                            with open(mp, encoding="utf-8") as f:
                                mode = json.load(f).get("control_mode", "")
                        except Exception:
                            mode = ""
                        runs.append({"label": d, "mode": mode})
                return self._send(200, json.dumps({"runs": runs}, ensure_ascii=False).encode("utf-8"),
                                  "application/json; charset=utf-8")
            if url.path == "/api/data":
                run = parse_qs(url.query).get("run", [""])[0]
                data = load_run(run)
                if data is None:
                    return self._send(404, b'{"error": "run not found"}', "application/json")
                return self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                                  "application/json; charset=utf-8")
            self._send(404, b"not found", "text/plain")
        except Exception as exc:   # 防止单次读文件异常打断轮询
            try:
                self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"),
                           "application/json")
            except Exception:
                pass

    def log_message(self, *args):   # 静默访问日志, 避免刷屏影响演示
        pass


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="功能二·实时数据可视化服务器(云端)")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[仪表盘] 已启动: http://{args.host}:{args.port}  (Ctrl+C停止)")
    print(f"[仪表盘] 数据目录: {DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[仪表盘] 已停止")


if __name__ == "__main__":
    main()
