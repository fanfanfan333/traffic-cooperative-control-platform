# -*- coding: utf-8 -*-
"""
ONNX Runtime 策略包装 (赛道C · 模型部署层)
=========================================
将 export_onnx.py 产出的 ONNX 模型包装成与 TrafficDQNAgent 相同的接口,
使 sumo_pipeline / data_collector 等既有链路无需改动即可切换到
"边缘部署形态" (ONNX Runtime 推理) 运行。

仅支持评估/部署推理, 不支持训练 (ONNX 模型不参与梯度更新)。
"""
import os

import numpy as np


class OnnxPolicyAgent:
    """与 TrafficDQNAgent 接口对齐的 ONNX 推理策略。"""

    def __init__(self, onnx_path: str, state_dim: int = None):
        import onnxruntime as ort
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path} "
                                    f"(请先运行 python export_onnx.py)")
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # 从模型元数据推断维度 (state: [batch, 61])
        self.state_dim = int(inp.shape[1]) if state_dim is None else state_dim
        out = self.session.get_outputs()[0]
        self.action_dim = int(out.shape[1])
        # 与 DQNAgent 对齐的只读属性 (供日志打印/CSV记录)
        self.epsilon = 0.0
        self.current_lr = 0.0
        print(f"[ONNX] 部署模型已加载: {onnx_path} "
              f"(state_dim={self.state_dim}, action_dim={self.action_dim}, CPU推理)")

    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q = self.session.run(None, {self.input_name: x})[0]
        return int(np.argmax(q[0]))

    # ── 以下为与训练接口对齐的空操作, 保证 run_episode 调用链完整 ──
    def store(self, *a, **kw):
        pass

    def train_step(self):
        return None

    def flush_replay(self):
        pass

    def save(self, path):
        raise NotImplementedError("ONNX 部署模型只读, 不支持保存训练权重")

    def load(self, path):
        raise NotImplementedError("OnnxPolicyAgent 构建时即加载, 不支持再加载 .pt")


def resolve_onnx_path(pt_path: str, explicit: str = None) -> str:
    """由 .pt 路径推导默认 ONNX 部署路径: model_deployment/<name>.onnx"""
    if explicit:
        return explicit
    base = os.path.splitext(os.path.basename(pt_path))[0]
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "model_deployment", base + ".onnx")
