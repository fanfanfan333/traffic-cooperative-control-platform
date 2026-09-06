# -*- coding: utf-8 -*-
"""
SUMO 状态：按「进口方向 × 转向」统计密度/排队/速度。

单路口: 4方向 × 3转向 × 3特征 + 4相位 one-hot + 1相位时长 = 41 维
多路口(干线): 再加 4 个方向槽位 × (邻居相位one-hot 4 + 邻居相位时长 1) = +20 维
统一为 61 维 —— 单路口的邻居槽位全为0, 单路口/干线可共用同一个网络。

方向识别采用 几何法: 按边的起点节点相对信号路口的方位判定进口方向,
不依赖边命名(E_/W_/link_X等任意命名均兼容, 干线路网的走廊边也能正确归并)。
"""
import numpy as np
import sumolib


DIR_TO_PREFIX = {'东': 'E', '西': 'W', '北': 'N', '南': 'S'}   # 兼容旧引用保留

PHASE_NAMES = {
    0: "南北向直行左转",
    1: "南北向左转",
    2: "东西向直行左转",
    3: "东西向左转",
}
TURN_LABELS = ("直行", "左转", "右转")
TURN_CODE = {'s': '直行', 'l': '左转', 'r': '右转'}
ENTRANCE_DIRS = ('东', '西', '北', '南')


def edge_approach_direction(net, node, edge):
    """几何法判定进口方向: 边起点节点在路口的哪一侧, 就从哪个方向进入路口。
    (W_in起点在路口西侧→'西'; 干线走廊边 link_0 起点是上游路口J0, 同样按方位判定)
    返回 东南西北 之一; 边起点就是路口自身时返回 None。"""
    frm = edge.getFromNode()
    if frm.getID() == node.getID():
        return None
    fx, fy = frm.getCoord()
    x, y = node.getCoord()
    dx, dy = fx - x, fy - y
    if abs(dx) >= abs(dy):
        return "西" if dx < 0 else "东"
    return "南" if dy < 0 else "北"


def build_tls_adjacency(net):
    """多路口协同的拓扑关系: {路口ID: {方向: 相邻路口ID}}。
    沿某方向的下一条边若通向另一个信号灯节点, 即为该方向的邻居路口 ——
    干线走廊上互为上下游, 是绿波协调的信息来源。"""
    adjacency = {}
    for node in net.getNodes():
        if node.getType() != "traffic_light":
            continue
        adjacency[node.getID()] = {}
        x, y = node.getCoord()
        for edge in node.getOutgoing():
            to_node = edge.getToNode()
            if to_node.getID() == node.getID() or to_node.getType() != "traffic_light":
                continue
            tx, ty = to_node.getCoord()
            dx, dy = tx - x, ty - y
            if abs(dx) >= abs(dy):
                d = "东" if dx > 0 else "西"
            else:
                d = "北" if dy > 0 else "南"
            # 同方向多个可达路口时保留最近的
            if d not in adjacency[node.getID()]:
                adjacency[node.getID()][d] = to_node.getID()
    return adjacency


class SumoStateBuilder:
    """为路网内每一个信号灯路口构建状态向量(多路口协同版)。
    build(tls_id, ...) 按路口逐个调用; 邻居相位上下文由 tls_phases 提供。"""

    def __init__(self, traci_conn, net_path: str, max_vehicles_per_group: int = 20,
                 max_speed: float = 13.89, max_phase_duration: float = 120.0,
                 tls_id: str = None):
        self.traci = traci_conn
        self.max_v = max_vehicles_per_group
        self.max_speed = max_speed
        self.max_phase_duration = max_phase_duration

        net = sumolib.net.readNet(net_path)
        self._net = net
        tls_nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
        if not tls_nodes:
            raise ValueError(f"{net_path} 中没有信号灯节点")
        # tls_id 仅作筛选(兼容旧调用); 默认覆盖路网内全部信号灯路口
        if tls_id is not None:
            picked = [n for n in tls_nodes if n.getID() == tls_id] or tls_nodes[:1]
            tls_nodes = picked
        self.tls_ids = [n.getID() for n in tls_nodes]
        self.tls_nodes = {n.getID(): n for n in tls_nodes}

        # 每个路口: 进口车道按 (几何方向 × 转向) 归组
        self.lanes_by_group = {}
        for tls in self.tls_ids:
            node = self.tls_nodes[tls]
            groups = {(d, t): [] for d in ENTRANCE_DIRS for t in TURN_LABELS}
            seen = {k: set() for k in groups}
            for conn in node.getConnections():
                from_edge = conn.getFrom()
                direction = edge_approach_direction(net, node, from_edge)
                turn = TURN_CODE.get(conn.getDirection())
                if direction is None or turn is None:
                    continue
                if hasattr(conn, "getFromLaneIndex"):
                    lane = from_edge.getLane(conn.getFromLaneIndex())
                else:
                    lane = conn.getFromLane()
                lane_id = lane.getID()
                key = (direction, turn)
                if lane_id not in seen[key]:
                    seen[key].add(lane_id)
                    groups[key].append(lane_id)
            self.lanes_by_group[tls] = groups

        # 邻居拓扑(绿波协调用)
        full_adj = build_tls_adjacency(net)
        self.adjacency = {tls: full_adj.get(tls, {}) for tls in self.tls_ids}

        # 状态维度: 自身41 + 4方向槽位×5邻居特征 (单路口邻居槽位恒为0)
        self.own_dim = len(ENTRANCE_DIRS) * len(TURN_LABELS) * 3 + len(PHASE_NAMES) + 1
        self.neighbor_dim = len(ENTRANCE_DIRS) * 5
        self.state_dim = self.own_dim + self.neighbor_dim

    def build(self, tls_id: str, current_phase: int, phase_duration_sec: float,
              tls_phases: dict = None) -> np.ndarray:
        """tls_phases: {路口ID: (相位, 相位已持续秒)}, 提供邻居路口的实时相位上下文。"""
        if tls_id not in self.lanes_by_group:
            raise KeyError(f"未知路口 {tls_id}, 可用: {self.tls_ids}")
        feats = []
        for direction in ENTRANCE_DIRS:
            for turn in TURN_LABELS:
                lanes = self.lanes_by_group[tls_id][(direction, turn)]
                if not lanes:
                    feats.extend([0.0, 0.0, 0.0])
                    continue
                veh_count, halt_count, speed_sum = 0, 0, 0.0
                for lane_id in lanes:
                    veh_count += self.traci.lane.getLastStepVehicleNumber(lane_id)
                    halt_count += self.traci.lane.getLastStepHaltingNumber(lane_id)
                    speed_sum += self.traci.lane.getLastStepMeanSpeed(lane_id)
                density = np.clip(veh_count / self.max_v, 0, 1)
                queue_len = np.clip(halt_count / self.max_v, 0, 1)
                avg_speed = np.clip((speed_sum / len(lanes)) / self.max_speed, 0, 1)
                feats.extend([density, queue_len, avg_speed])

        phase_onehot = np.zeros(len(PHASE_NAMES), dtype=np.float32)
        phase_onehot[current_phase] = 1.0
        phase_norm = np.clip(phase_duration_sec / self.max_phase_duration, 0, 1)

        # ── 邻居路口相位上下文: 干线协同的关键输入 ──
        # 上游刚放行的车队约36秒后抵达(500m/13.9m/s), 本路口据此提前腾空绿灯;
        # 单路口无邻居, 该段恒为0。
        neigh = []
        my_adj = self.adjacency.get(tls_id, {})
        for direction in ENTRANCE_DIRS:
            nb = my_adj.get(direction)
            if nb and tls_phases and nb in tls_phases:
                ph, dur = tls_phases[nb]
                onehot = np.zeros(len(PHASE_NAMES), dtype=np.float32)
                onehot[ph] = 1.0
                neigh.extend(onehot.tolist())
                neigh.append(np.clip(dur / self.max_phase_duration, 0, 1))
            else:
                neigh.extend([0.0] * 5)

        return np.concatenate([
            np.array(feats, dtype=np.float32),
            phase_onehot,
            np.array([phase_norm], dtype=np.float32),
            np.array(neigh, dtype=np.float32),
        ])
