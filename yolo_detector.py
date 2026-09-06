"""
YOLOv8 目标检测封装器
集成 Ultralytics YOLOv8，专为交通场景优化
输出格式对接 DeepSORT 追踪器
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[Warning] ultralytics 未安装，请运行 pip install ultralytics")


# 交通场景感兴趣类别（COCO 类别 ID）
TRAFFIC_CLASS_IDS = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    0:  "person",       # 行人
    1:  "bicycle",
}

TRAFFIC_CLASS_NAMES = {v: k for k, v in TRAFFIC_CLASS_IDS.items()}


@dataclass
class Detection:
    bbox:     np.ndarray      # [x1, y1, x2, y2]  xyxy 格式
    score:    float
    class_id: int
    class_name: str = ""

    def to_dict(self) -> dict:
        return {
            "bbox"      : self.bbox.tolist(),
            "score"     : self.score,
            "class_id"  : self.class_id,
            "class_name": self.class_name,
        }


class YOLOv8Detector:
    """
    YOLOv8 检测器封装
    支持置信度过滤、NMS、ROI 裁剪、类别白名单
    """

    def __init__(
        self,
        model_path:    str   = "yolov8n.pt",   # nano / small / medium / large / xlarge
        conf_thresh:   float = 0.35,
        iou_thresh:    float = 0.45,            # NMS IoU 阈值
        img_size:      int   = 640,
        device:        str   = "cpu",           # "cpu" | "cuda" | "mps"
        class_filter:  Optional[list] = None,   # None = 全部, list[int] = 只检测这些类
        roi:           Optional[list] = None,   # [x1, y1, x2, y2] 感兴趣区域（像素）
    ):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("请先安装 ultralytics: pip install ultralytics")

        self.model       = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.img_size    = img_size
        self.device      = device
        self.class_filter = class_filter if class_filter else list(TRAFFIC_CLASS_IDS.keys())
        self.roi         = roi   # 可选：只检测画面中的指定区域

        print(f"[YOLOv8] 模型加载完成: {model_path} | device={device} | conf={conf_thresh}")

    def _apply_roi(self, frame: np.ndarray):
        """裁切 ROI 区域，同时返回偏移量用于坐标还原"""
        if self.roi is None:
            return frame, 0, 0
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2], x1, y1

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        对单帧执行目标检测

        Args:
            frame: BGR 格式 numpy 数组

        Returns:
            List[Detection]，按置信度降序排列
        """
        crop, off_x, off_y = self._apply_roi(frame)

        results = self.model.predict(
            source    = crop,
            conf      = self.conf_thresh,
            iou       = self.iou_thresh,
            imgsz     = self.img_size,
            device    = self.device,
            classes   = self.class_filter,
            verbose   = False,
        )

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy     = box.xyxy[0].cpu().numpy()
                score    = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # 坐标还原（ROI 偏移）
                xyxy[0] += off_x
                xyxy[2] += off_x
                xyxy[1] += off_y
                xyxy[3] += off_y

                detections.append(Detection(
                    bbox      = xyxy,
                    score     = score,
                    class_id  = class_id,
                    class_name= TRAFFIC_CLASS_IDS.get(class_id, "unknown"),
                ))

        # 按置信度降序
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """批量推理（提高 GPU 利用率）"""
        results_all = self.model.predict(
            source  = frames,
            conf    = self.conf_thresh,
            iou     = self.iou_thresh,
            imgsz   = self.img_size,
            device  = self.device,
            classes = self.class_filter,
            verbose = False,
        )
        batch_dets = []
        for r in results_all:
            dets = []
            if r.boxes:
                for box in r.boxes:
                    xyxy     = box.xyxy[0].cpu().numpy()
                    score    = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    dets.append(Detection(
                        bbox=xyxy, score=score, class_id=class_id,
                        class_name=TRAFFIC_CLASS_IDS.get(class_id, "unknown"),
                    ))
            dets.sort(key=lambda d: d.score, reverse=True)
            batch_dets.append(dets)
        return batch_dets
