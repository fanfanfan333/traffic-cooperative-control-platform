"""
交通预测流主管线
YOLO8 检测 → DeepSORT 追踪 → 状态构建 → DQN 信号控制

使用方法:
    python pipeline.py --source video.mp4 --yolo yolov8n.pt
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path

from yolo_detector import YOLOv8Detector
from deepsort_tracker import DeepSORT
from traffic_dqn import (
    TrafficStateBuilder,
    RewardCalculator,
    TrafficDQNAgent,
)


# ─────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────

PALETTE = {
    0: (0, 255, 255),    # person  → 黄
    1: (255, 200, 0),    # bicycle → 蓝黄
    2: (0, 128, 255),    # car     → 橙
    3: (255, 0, 128),    # moto    → 粉
    5: (0, 255, 0),      # bus     → 绿
    7: (255, 0, 0),      # truck   → 蓝
}

PHASE_LABELS = {
    0: "NS-Straight  ↑↓",
    1: "NS-Left-Turn ↰↱",
    2: "EW-Straight  ←→",
    3: "EW-Left-Turn ↲↳",
}


def draw_tracks(frame, tracks):
    for trk in tracks:
        x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
        tid     = trk["track_id"]
        cid     = trk["class_id"]
        vx, vy  = trk["velocity"]
        color   = PALETTE.get(cid, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} v=({vx:.1f},{vy:.1f})"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_hud(frame, phase, phase_dur, action, reward, epsilon, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"Phase   : {PHASE_LABELS.get(phase, phase)} ({phase_dur}f)",
        f"Action  : {PHASE_LABELS.get(action, action)}",
        f"Reward  : {reward:+.3f}",
        f"Epsilon : {epsilon:.4f}",
        f"FPS     : {fps:.1f}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 22 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ─────────────────────────────────────────────
# 主管线
# ─────────────────────────────────────────────

class TrafficPipeline:
    def __init__(self, args):
        # ── 检测器 ──
        self.detector = YOLOv8Detector(
            model_path  = args.yolo,
            conf_thresh = args.conf,
            iou_thresh  = args.iou,
            img_size    = args.imgsz,
            device      = args.device,
        )

        # ── 追踪器 ──
        self.tracker = DeepSORT(
            reid_model_path   = args.reid,
            max_age           = 30,
            n_init            = 3,
            appearance_thresh = 0.4,
            iou_thresh        = 0.7,
        )

        # ── DQN 状态/奖励/智能体 ──
        self.state_builder = TrafficStateBuilder(
            frame_w=args.width, frame_h=args.height
        )
        self.reward_calc = RewardCalculator()
        self.agent = TrafficDQNAgent(
            state_dim   = self.state_builder.state_dim,
            action_dim  = 4,
            epsilon     = 1.0 if not args.eval else 0.0,
            lr          = args.lr,
            n_step      = args.n_step,
        )
        if args.model:
            if not Path(args.model).is_file():
                print(f"[Pipeline] 指定的模型文件不存在，跳过加载，将从头训练: {args.model}")
            else:
                try:
                    self.agent.load(args.model)
                    print(f"[Pipeline] 已加载模型: {args.model}")
                except Exception as exc:
                    print(f"[Pipeline] 模型加载失败，跳过并从头训练。原因: {exc}")

        # ── 信号状态 ──
        self.current_phase    = 0
        self.phase_duration   = 0
        self.min_phase_frames = args.min_green    # 最小绿灯帧数
        self.max_phase_frames = args.max_green    # 最大绿灯帧数
        self.current_state: np.ndarray = np.zeros(self.state_builder.state_dim)
        self.last_reward = 0.0

        self.args = args
        self._train_mode = not args.eval

    def run(self):
        cap = cv2.VideoCapture(self.args.source if self.args.source else 0)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.args.source}")

        out = None
        if self.args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self.args.save, fourcc, 25,
                (int(cap.get(3)), int(cap.get(4)))
            )

        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. 目标检测 ──
            detections = self.detector.detect(frame)
            det_dicts  = [d.to_dict() for d in detections]

            # ── 2. 多目标追踪 ──
            tracks = self.tracker.update(frame, det_dicts)

            # ── 3. 构建 DQN 状态 ──
            next_state = self.state_builder.build(
                tracks         = tracks,
                current_phase  = self.current_phase,
                phase_duration = self.phase_duration,
            )

            # ── 4. 计算奖励 ──
            throughput = self._estimate_throughput(tracks)
            reward = self.reward_calc.compute(
                state      = self.current_state,
                next_state = next_state,
                throughput = throughput,
            )
            self.last_reward = reward

            # ── 5. DQN 决策（每帧都询问，但受最小绿灯约束）──
            action = self.agent.select_action(
                next_state, greedy=not self._train_mode
            )

            # 最小绿灯保护
            if self.phase_duration < self.min_phase_frames:
                action = self.current_phase   # 强制维持当前相位
            elif self.phase_duration >= self.max_phase_frames:
                action = (self.current_phase + 1) % 4  # 强制切换

            # ── 6. 存储经验 & 训练 ──
            if self._train_mode:
                # 每 500 帧视为 episode 结束(frame 0 不算,否则第0帧就会被
                # 当成终态、这一条经验的bootstrap被错误截断)
                done = frame_idx > 0 and frame_idx % 500 == 0
                self.agent.store(
                    self.current_state, action,
                    reward, next_state, done
                )
                loss = self.agent.train_step()

            # ── 7. 执行动作（更新信号相位）──
            if action != self.current_phase:
                self.current_phase  = action
                self.phase_duration = 0
            else:
                self.phase_duration += 1

            self.current_state = next_state

            # ── 8. 可视化 ──
            fps = (frame_idx + 1) / (time.time() - t0 + 1e-6)
            draw_tracks(frame, tracks)
            draw_hud(frame, self.current_phase, self.phase_duration,
                     action, reward, self.agent.epsilon, fps)

            if out:
                out.write(frame)
            if self.args.display:
                cv2.imshow("Traffic Prediction Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"[Frame {frame_idx:5d}] tracks={len(tracks):2d} "
                      f"phase={self.current_phase} ε={self.agent.epsilon:.4f} "
                      f"reward={reward:+.3f} fps={fps:.1f}")

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        if self._train_mode and self.args.model:
            # 视频结束时把n-step滑动窗口里剩余的经验也冲进回放池再保存
            self.agent.flush_replay()
            self.agent.save(self.args.model)
            print(f"[Pipeline] 训练完成，模型已保存")

    def _estimate_throughput(self, tracks: list[dict]) -> int:
        """
        简单估计吞吐量：统计当前帧速度足够大且接近路口中心的车辆
        （实际可用虚拟检测线计数）
        """
        count = 0
        for trk in tracks:
            vx, vy = trk["velocity"]
            speed = np.sqrt(vx**2 + vy**2)
            cx = (trk["bbox"][0] + trk["bbox"][2]) / 2 / self.args.width
            cy = (trk["bbox"][1] + trk["bbox"][3]) / 2 / self.args.height
            # 通过路口中心区域 (0.4~0.6) 且速度较快
            if 0.4 < cx < 0.6 and 0.4 < cy < 0.6 and speed > 5:
                count += 1
        return count


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="交通预测流：YOLO8 + DeepSORT + DQN")
    p.add_argument("--source",    default=0,             help="视频路径或摄像头 ID")
    p.add_argument("--yolo",      default="yolov8n.pt",  help="YOLOv8 权重路径")
    p.add_argument("--reid",      default=None,          help="ReID ONNX 模型路径")
    p.add_argument("--model",     default=None, help="DQN 模型保存/加载路径（不填则不加载，从头训练）")
    p.add_argument("--conf",      default=0.35, type=float)
    p.add_argument("--iou",       default=0.45, type=float)
    p.add_argument("--imgsz",     default=640,  type=int)
    p.add_argument("--width",     default=1280, type=int,  help="帧宽（用于坐标归一化）")
    p.add_argument("--height",    default=720,  type=int,  help="帧高")
    p.add_argument("--device",    default="cpu",           help="cpu / cuda / mps")
    p.add_argument("--min-green", default=30,  type=int,   help="最小绿灯帧数")
    p.add_argument("--max-green", default=150, type=int,   help="最大绿灯帧数")
    p.add_argument("--save",      default=None,            help="结果视频保存路径")
    p.add_argument("--display",   action="store_true",     help="实时显示窗口")
    p.add_argument("--eval",      action="store_true",     help="评估模式（不训练，ε=0）")
    p.add_argument("--lr",        default=3e-3, type=float, help="DQN学习率(默认3e-3,配合warmup+余弦退火)")
    p.add_argument("--n-step",    default=3,    type=int,   help="n-step经验回放视野(默认3,设1退化为1-step)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrafficPipeline(args)
    pipeline.run()