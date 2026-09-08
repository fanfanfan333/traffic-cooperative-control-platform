# -*- coding: utf-8 -*-
"""
生成平台 REST API 的 Postman Collection (v2.1) / Apifox 导入文件
对应赛题功能一任务1: "使用工具(如Postman、Apifox)定义并模拟测试关键的数据请求/响应接口"

用法:
    python gen_postman_collection.py
产物:
    platform/XH-202613_platform_api.postman_collection.json
    (Postman: Import → 选择文件; Apifox: 导入 → Postman 格式)
说明:
    接口清单与 platform/app.py 的 /api/docs 保持一致, 新增接口时请同步更新本文件。
"""
import json
import os

BASE_URL = "http://127.0.0.1:5000"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "platform", "XH-202613_platform_api.postman_collection.json")

# (分组, 方法, 路径, 说明, 请求体示例 or None)
APIS = [
    ("总览与日志", "GET", "/api/overview", "总览KPI/系统资源/训练仿真评估状态/实时趋势", None),
    ("总览与日志", "GET", "/api/logs", "系统操作日志", None),
    ("路网与场景", "GET", "/api/intersections", "全部路口与干线场景清单(含雄安实测溯源/生成场景)", None),
    ("路网与场景", "POST", "/api/scenario/generate", "扰动场景生成(早高峰/平峰/突发事件/随机)",
     {"base": "intersection_1", "kind": "早高峰"}),
    ("算法训练", "POST", "/api/train/start", "启动DQN训练",
     {"episodes": 10, "model": "dqn_unified.pt", "scenario": "all"}),
    ("算法训练", "POST", "/api/train/stop", "停止训练", None),
    ("算法训练", "GET", "/api/train/status", "训练状态/日志尾部/最新回合指标", None),
    ("算法训练", "GET", "/api/train/history", "训练历史曲线数据", None),
    ("信号仿真", "POST", "/api/sim/start", "启动信号实时仿真",
     {"scenario": "intersection_1", "policy": "dqn", "seconds": 600}),
    ("信号仿真", "POST", "/api/sim/stop", "停止仿真", None),
    ("信号仿真", "GET", "/api/sim/state", "仿真实时状态(相位/指标/曲线)", None),
    ("性能评估", "POST", "/api/eval/run", "启动性能评估(固定配时vs DQN)",
     {"scenario": "intersection_1", "seconds": 900, "seed": 0}),
    ("性能评估", "GET", "/api/eval/status", "评估进度与六维雷达结果", None),
    ("云边端与部署", "GET", "/api/edge", "云边端三层实时统计", None),
    ("云边端与部署", "GET", "/api/deploy", "模型转换与轻量化部署基准(ONNX/FP16/INT8/精度一致性)", None),
    ("采集数据", "GET", "/api/runs", "采集数据运行清单", None),
    ("采集数据", "GET", "/api/data?run=dqn", "采集数据明细(降采样)", None),
]


def main():
    groups = {}
    for grp, method, path, desc, body in APIS:
        item = {
            "name": f"{method} {path}",
            "request": {
                "method": method,
                "header": [{"key": "Content-Type", "value": "application/json"}] if method == "POST" else [],
                "url": {"raw": "{{base_url}}" + path.split("?")[0],
                        "host": ["{{base_url}}"],
                        "path": [p for p in path.split("?")[0].split("/") if p],
                        **({"query": [{"key": kv.split("=")[0], "value": kv.split("=")[1]}
                                       for kv in path.split("?")[1].split("&")]} if "?" in path else {})},
                "description": desc,
            },
            "response": [],
        }
        if body is not None:
            item["request"]["body"] = {"mode": "raw", "raw": json.dumps(body, ensure_ascii=False, indent=2),
                                       "options": {"raw": {"language": "json"}}}
        groups.setdefault(grp, []).append(item)

    collection = {
        "info": {
            "_postman_id": "xh-202613-platform-api",
            "name": "XH-202613 车路云一体化协同管控平台 API",
            "description": "面向雄安新区城市大脑的车路云一体化协同管控算法与仿真平台 REST 接口集合。"
                           "启动平台: python platform/app.py --port 5000",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "base_url", "value": BASE_URL}],
        "item": [{"name": grp, "item": items} for grp, items in groups.items()],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)
    n = sum(len(v) for v in groups.values())
    print(f"[Postman] 集合已生成: {OUT} ({len(groups)} 组, {n} 个接口)")


if __name__ == "__main__":
    main()
