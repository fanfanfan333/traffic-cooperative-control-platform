"""
DeepSORT 多目标追踪器
集成卡尔曼滤波、匈牙利匹配算法、ReID特征提取
用于交通预测流中的车辆追踪
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from collections import deque
import cv2


# ─────────────────────────────────────────────
# 1. 卡尔曼滤波器（运动模型）
# ─────────────────────────────────────────────

class KalmanFilter:
    """
    恒速运动模型的卡尔曼滤波器
    状态向量: [cx, cy, w, h, vx, vy, vw, vh]
      cx, cy: 边界框中心坐标
      w, h  : 宽高
      vx, vy, vw, vh: 对应速度
    """

    def __init__(self):
        self.dim_x = 8   # 状态维度
        self.dim_z = 4   # 观测维度 [cx, cy, w, h]

        # 状态转移矩阵 F（匀速模型）
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0   # cx += vx * dt (dt=1)

        # 观测矩阵 H
        self.H = np.eye(self.dim_z, self.dim_x, dtype=np.float32)

        # 过程噪声协方差 Q
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # 观测噪声协方差 R
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[2:, 2:] *= 10.0  # 宽高测量噪声更大

        # 初始协方差 P
        self.P = np.eye(self.dim_x, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0  # 速度初始不确定性大

    def initiate(self, measurement: np.ndarray):
        """由初始检测 [cx, cy, w, h] 初始化状态"""
        x = np.zeros(self.dim_x, dtype=np.float32)
        x[:4] = measurement
        P = self.P.copy()
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray):
        """预测步骤"""
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray):
        """更新步骤（EKF 线性化省略，直接 KF）"""
        y = z - self.H @ x                         # 创新量
        S = self.H @ P @ self.H.T + self.R         # 创新协方差
        K = P @ self.H.T @ np.linalg.inv(S)        # 卡尔曼增益
        x = x + K @ y
        P = (np.eye(self.dim_x) - K @ self.H) @ P
        return x, P


# ─────────────────────────────────────────────
# 2. ReID 特征提取（轻量 CNN）
# ─────────────────────────────────────────────

class ReIDFeatureExtractor:
    """
    轻量 ReID 特征提取器
    实际项目可替换为 OSNet / FastReID / BoT-SORT 的特征网络
    这里用 OpenCV DNN 加载预训练模型作为示例骨架
    """

    def __init__(self, model_path: str = None, input_size=(64, 128)):
        self.input_size = input_size   # (W, H)
        self.net = None

        if model_path:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, self.input_size)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img.transpose(2, 0, 1)[np.newaxis]   # (1, C, H, W)

    def extract(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """
        从原始帧中裁切目标区域，提取 ReID 特征向量

        Args:
            frame    : BGR 帧图像
            bbox_xyxy: [x1, y1, x2, y2]
        Returns:
            feature  : L2 归一化特征向量 (128,)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        if self.net is not None:
            blob = self._preprocess(crop)
            self.net.setInput(blob)
            feat = self.net.forward().flatten()
        else:
            # Fallback：用颜色直方图作为伪特征（仅开发调试用）
            feat = self._color_histogram(crop)

        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-6)

    @staticmethod
    def _color_histogram(crop: np.ndarray, bins=32) -> np.ndarray:
        hist = []
        for ch in range(3):
            h, _ = np.histogram(crop[:, :, ch], bins=bins, range=(0, 256))
            hist.append(h.astype(np.float32))
        feat = np.concatenate(hist)  # (96,) → pad to (128,)
        feat = np.pad(feat, (0, 128 - len(feat)))
        return feat


# ─────────────────────────────────────────────
# 3. 单目标轨迹 Track
# ─────────────────────────────────────────────

class TrackState:
    Tentative  = 1   # 候选（尚未确认）
    Confirmed  = 2   # 已确认
    Deleted    = 3   # 待删除


class Track:
    _next_id = 1

    def __init__(self, bbox_xyxy: np.ndarray, score: float,
                 class_id: int, feature: np.ndarray,
                 max_age: int = 30, n_init: int = 3):
        self.track_id   = Track._next_id
        Track._next_id += 1

        self.class_id   = class_id
        self.score      = score
        self.state      = TrackState.Tentative
        self.hits       = 1
        self.age        = 1
        self.time_since_update = 0
        self.max_age    = max_age
        self.n_init     = n_init

        # 历史外观特征（EMA 平滑）
        self.features: deque = deque(maxlen=10)
        self.features.append(feature)

        # 卡尔曼滤波初始化
        self.kf = KalmanFilter()
        meas = self._xyxy_to_cxcywh(bbox_xyxy)
        self.kf_state, self.kf_cov = self.kf.initiate(meas)

        # 速度历史（用于 DQN 状态构建）
        self.velocity_history: deque = deque(maxlen=10)

    # ---- bbox 转换工具 ----
    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)

    @staticmethod
    def _cxcywh_to_xyxy(state):
        cx, cy, w, h = state[:4]
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=np.float32)

    # ---- 对外接口 ----
    def predict(self):
        self.kf_state, self.kf_cov = self.kf.predict(self.kf_state, self.kf_cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray, score: float, feature: np.ndarray):
        meas = self._xyxy_to_cxcywh(bbox_xyxy)
        self.kf_state, self.kf_cov = self.kf.update(self.kf_state, self.kf_cov, meas)

        self.features.append(feature)
        self.score = score
        self.hits += 1
        self.time_since_update = 0

        # 记录速度向量（vx, vy）
        vx, vy = self.kf_state[4], self.kf_state[5]
        self.velocity_history.append((vx, vy))

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def get_bbox(self) -> np.ndarray:
        return self._cxcywh_to_xyxy(self.kf_state)

    def get_mean_feature(self) -> np.ndarray:
        return np.mean(list(self.features), axis=0)

    def get_velocity(self):
        if self.velocity_history:
            return self.velocity_history[-1]
        return (0.0, 0.0)

    def is_confirmed(self)  -> bool: return self.state == TrackState.Confirmed
    def is_deleted(self)    -> bool: return self.state == TrackState.Deleted
    def is_tentative(self)  -> bool: return self.state == TrackState.Tentative


# ─────────────────────────────────────────────
# 4. 匹配代价矩阵 & 匈牙利算法
# ─────────────────────────────────────────────

def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算两组 bbox 间的 IoU 矩阵 (M x N)"""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-6)


def appearance_cost(tracks: list, detections_feat: np.ndarray) -> np.ndarray:
    """余弦距离外观代价矩阵 (T x D)"""
    track_feats = np.array([t.get_mean_feature() for t in tracks])  # (T, feat_dim)
    # cosine 距离 = 1 - 余弦相似度
    cost = np.array([
        [cosine(tf, df) for df in detections_feat]
        for tf in track_feats
    ], dtype=np.float32)
    return cost


def hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """
    匈牙利算法匹配，过滤代价超过 threshold 的对
    Returns: (matched_pairs, unmatched_tracks_idx, unmatched_dets_idx)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched, unmatched_t, unmatched_d = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            unmatched_t.append(r)
            unmatched_d.append(c)
        else:
            matched.append((r, c))

    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_t.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in col_ind:
            unmatched_d.append(c)

    return matched, list(set(unmatched_t)), list(set(unmatched_d))


# ─────────────────────────────────────────────
# 5. DeepSORT 主追踪器
# ─────────────────────────────────────────────

class DeepSORT:
    """
    DeepSORT 多目标追踪器
    两阶段匹配：先外观（ReID）再 IoU
    """

    def __init__(
        self,
        reid_model_path: str = None,
        max_age:          int   = 30,
        n_init:           int   = 3,
        appearance_thresh: float = 0.4,
        iou_thresh:        float = 0.7,
        lambda_iou:        float = 0.4,   # 融合权重
    ):
        self.tracks: list[Track] = []
        self.extractor = ReIDFeatureExtractor(reid_model_path)
        self.max_age   = max_age
        self.n_init    = n_init
        self.app_thresh = appearance_thresh
        self.iou_thresh = iou_thresh
        self.lambda_iou = lambda_iou

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],   # [{"bbox": [x1,y1,x2,y2], "score": f, "class_id": i}, ...]
    ) -> list[dict]:
        """
        执行一帧的追踪更新

        Returns:
            list of {"track_id", "bbox", "class_id", "score", "velocity"}
        """
        # ── 1. 卡尔曼预测 ──
        for t in self.tracks:
            t.predict()

        if not detections:
            for t in self.tracks:
                t.mark_missed()
            self._remove_deleted()
            return self._output()

        det_bboxes = np.array([d["bbox"]     for d in detections], dtype=np.float32)
        det_scores = np.array([d["score"]    for d in detections], dtype=np.float32)
        det_cids   = [d["class_id"] for d in detections]

        # ── 2. ReID 特征提取 ──
        det_features = np.array([
            self.extractor.extract(frame, b) for b in det_bboxes
        ])

        # ── 3. 分离已确认 / 候选轨迹 ──
        confirmed_idx  = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_idx  = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        # ── 4a. 已确认轨迹：外观 + IoU 融合代价 ──
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = []

        if confirmed_idx:
            conf_tracks = [self.tracks[i] for i in confirmed_idx]
            conf_bboxes = np.array([t.get_bbox() for t in conf_tracks])

            app_cost = appearance_cost(conf_tracks, det_features)
            iou_mat  = iou_batch(conf_bboxes, det_bboxes)
            iou_cost = 1.0 - iou_mat

            fused_cost = (1 - self.lambda_iou) * app_cost + self.lambda_iou * iou_cost

            matched_c, unm_t_c, unm_d_c = hungarian_match(fused_cost, self.app_thresh)

            for ri, di in matched_c:
                self.tracks[confirmed_idx[ri]].update(
                    det_bboxes[di], det_scores[di], det_features[di]
                )
            unmatched_trks += [confirmed_idx[r] for r in unm_t_c]
            unmatched_dets  = unm_d_c

        # ── 4b. 候选轨迹：纯 IoU 匹配 ──
        if tentative_idx and unmatched_dets:
            tent_tracks = [self.tracks[i] for i in tentative_idx]
            tent_bboxes = np.array([t.get_bbox() for t in tent_tracks])
            rem_bboxes  = det_bboxes[unmatched_dets]

            iou_cost2 = 1.0 - iou_batch(tent_bboxes, rem_bboxes)
            matched_t, unm_t_t, unm_d_t = hungarian_match(iou_cost2, 1 - self.iou_thresh)

            for ri, di_local in matched_t:
                di = unmatched_dets[di_local]
                self.tracks[tentative_idx[ri]].update(
                    det_bboxes[di], det_scores[di], det_features[di]
                )
            unmatched_trks += [tentative_idx[r] for r in unm_t_t]
            unmatched_dets  = [unmatched_dets[d] for d in unm_d_t]

        # ── 5. 未匹配轨迹标记 / 新建轨迹 ──
        for idx in unmatched_trks:
            self.tracks[idx].mark_missed()

        for di in unmatched_dets:
            self.tracks.append(Track(
                det_bboxes[di], det_scores[di], det_cids[di], det_features[di],
                max_age=self.max_age, n_init=self.n_init
            ))

        self._remove_deleted()
        return self._output()

    def _remove_deleted(self):
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _output(self) -> list[dict]:
        results = []
        for t in self.tracks:
            if t.is_confirmed():
                vx, vy = t.get_velocity()
                results.append({
                    "track_id" : t.track_id,
                    "bbox"     : t.get_bbox().tolist(),
                    "class_id" : t.class_id,
                    "score"    : float(t.score),
                    "velocity" : (float(vx), float(vy)),
                })
        return results
