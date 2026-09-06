# -*- coding: utf-8 -*-
"""
SUMO / TraCI 环境封装 —— 替代原pipeline.py里"视频帧循环"的部分。

负责:
    1. 管理仿真步进(traci.simulationStep)
    2. 把DQN的离散动作(0~3)转换成信号灯真实的state字符串下发
    3. 处理最小/最大绿灯时间(单位:仿真秒,而不是视频帧数)
    4. 用TraCI提供的真实数据计算奖励(等待时间、吞吐量),
       比原来从状态向量里近似估算更准确
"""
import numpy as np
import sumolib

from sumo_state_builder import PHASE_NAMES, edge_approach_direction, TURN_CODE


def load_link_info(net, node):
    """读取信号灯每个link的(索引, 进口方向, 转向)列表。
    进口方向用几何法判定(见 sumo_state_builder.edge_approach_direction),
    不依赖边命名 —— 干线路网的走廊边 link_X 同样能正确归并。
    旧版按边名前缀映射, 遇到 link_0 等命名会得到 None, 导致走廊方向
    全程红灯(与右转bug同类, 曾造成干线0/800辆完成)。"""
    conns = sorted(node.getConnections(), key=lambda c: c.getTLLinkIndex())
    link_info = []
    for c in conns:
        entrance_dir = edge_approach_direction(net, node, c.getFrom())
        turn_label = TURN_CODE.get(c.getDirection())
        link_info.append((c.getTLLinkIndex(), entrance_dir, turn_label))
    return link_info


def build_state_string(phase_name: str, link_info: list) -> str:
    dirs_in_name = [d for d in ['东', '西', '北', '南'] if d in phase_name]
    mixed = '直行' in phase_name and '左转' in phase_name   # 混合放行相位
    left_only = '左转' in phase_name and '直行' not in phase_name  # 左转保护相位
    if not mixed and not left_only and '直行' not in phase_name and '左转' not in phase_name:
        raise ValueError(f"无法识别的相位名: {phase_name}")
    state = []
    for _, entrance_dir, turn_label in link_info:
        if entrance_dir not in dirs_in_name:
            state.append('r')
            continue
        if turn_label in ('直行', '右转'):
            # 直行相位(含混合相位)放行直行+右转; 右转跟随本方向直行放行,
            # 与原始固定配时的右转处理一致。旧版右转在所有相位全程红灯,
            # 右转车流永远过不去、在进口道无限堆积,是延误爆炸的主因(环境bug)。
            state.append('G' if '直行' in phase_name else 'r')
        elif turn_label == '左转':
            if mixed:
                state.append('g')      # 让行绿: 避让对向直行后通过(同原始配时)
            elif left_only:
                state.append('G')      # 保护绿: 左转专用相位
            else:
                state.append('r')
        else:
            state.append('r')
    return ''.join(state)


def yellow_state(green_state: str) -> str:
    # 'g'(让行绿)在黄灯期间保持不变,与原始固定配时一致
    # (原net黄灯相位如 yyyg: 左转让行绿延续)。
    return ''.join('y' if ch == 'G' else ('g' if ch == 'g' else 'r') for ch in green_state)


def all_red_state(green_state: str) -> str:
    return 'r' * len(green_state)


class SumoTrafficEnv:
    """
    SUMO 仿真环境(多信号灯版) —— 单路口是 tls_ids 长度=1 的特例, 行为完全一致。
    action 语义对齐 TrafficDQNAgent.ACTION_SPACE:
        0: 南北向直行左转   1: 南北向左转   2: 东西向直行左转   3: 东西向左转
    多路口(干线)用法: apply_actions({tls_id: action}) 一次驱动全部路口,
    各路口同时黄灯切换/同时绿灯保持; compute_reward() 按路口分别返回奖励,
    共享策略对每个路口各存一条经验, 配合状态里的邻居相位上下文实现绿波协同。
    """

    def __init__(self, traci_conn, net_path: str, tls_id: str = None,
                 yellow_duration: int = 3, all_red_duration: int = 2,
                 min_green_sec: int = 10, max_green_sec: int = 60,
                 max_sim_time: float = None,
                 reward_alpha: float = 5.0, reward_delay_coef: float = 5.0,
                 reward_gamma: float = 2.0, reward_clip: float = 30.0,
                 reward_fairness_coef: float = 5.0,
                 reward_wait_scale: float = 60.0, reward_delta_scale: float = 5.0,
                 calib_wait_scale: float = None, calib_delta_scale: float = None,
                 step_callback=None):
        self.traci = traci_conn
        net = sumolib.net.readNet(net_path)
        self._net = net
        tls_nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
        if not tls_nodes:
            raise ValueError(f"{net_path} 中没有信号灯节点")
        if tls_id is not None:   # 兼容旧的单路口显式指定
            picked = [n for n in tls_nodes if n.getID() == tls_id] or tls_nodes[:1]
            tls_nodes = picked
        self.tls_ids = [n.getID() for n in tls_nodes]
        self._tls_nodes = {n.getID(): n for n in tls_nodes}
        self.link_info = {t: load_link_info(net, self._tls_nodes[t]) for t in self.tls_ids}
        self.green_states = {
            t: {action: build_state_string(name, self.link_info[t])
                for action, name in PHASE_NAMES.items()}
            for t in self.tls_ids
        }
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        self.min_green_sec = min_green_sec
        self.max_green_sec = max_green_sec
        self.max_sim_time = max_sim_time
        self.hold_sec = min_green_sec
        self.reward_alpha = reward_alpha
        self.reward_delay_coef = reward_delay_coef
        self.reward_gamma = reward_gamma
        self.reward_clip = reward_clip
        self.reward_fairness_coef = reward_fairness_coef
        self.reward_wait_scale = reward_wait_scale
        self.reward_delta_scale = reward_delta_scale
        # 每个仿真步都会调用的回调(数据采集模块的挂接点, 见 data_collector.py)
        self.step_callback = step_callback

        # 逐路口状态(干线协同: 各路口独立的相位/时长/等待基准)
        self.current_phase = {t: 0 for t in self.tls_ids}
        self.phase_duration_sec = {t: 0 for t in self.tls_ids}
        self._prev_avg_waiting = {t: 0.0 for t in self.tls_ids}
        self._last_mean_wait = {t: 0.0 for t in self.tls_ids}
        self._last_elapsed = 1
        self._last_arrived = 0

        # calib_wait_scale/calib_delta_scale: 跨episode慢慢校准出来的
        # "该路口典型严重程度"参考值,由调用方在多个episode之间持久化维护。
        # 本episode内保持固定不变。多路口时各路口共用调用方传入的该场景校准值。
        self._fixed_wait_scale = max(reward_wait_scale, calib_wait_scale or 0.0)
        self._fixed_delta_scale = max(reward_delta_scale, calib_delta_scale or 0.0)
        self._episode_wait_sum = 0.0
        self._episode_wait_max = 0.0
        self._episode_delta_abs_sum = 0.0
        self._episode_sample_count = 0

    def reset(self):
        for tls in self.tls_ids:
            self.current_phase[tls] = 0
            self.phase_duration_sec[tls] = 0.0
            self.traci.trafficlight.setRedYellowGreenState(
                self.tls_id_of(tls), self.green_states[tls][0]
            )
            total_waiting, veh_count = self._get_total_waiting_time_and_count(tls)
            self._prev_avg_waiting[tls] = total_waiting / max(veh_count, 1)
            self._last_mean_wait[tls] = self._prev_avg_waiting[tls]

    def tls_id_of(self, tls):
        return tls   # 语义占位: 环境ID即TraCI的tls ID

    def _step_n(self, n: int):
        arrived = 0
        elapsed = 0
        for _ in range(max(n, 0)):
            if self.is_finished():
                break
            self.traci.simulationStep()
            if self.step_callback is not None:
                self.step_callback()
            arrived += self.traci.simulation.getArrivedNumber()
            elapsed += 1
        return elapsed, arrived

    def apply_actions(self, actions: dict):
        """一次决策驱动全部信号灯。需要切换的路口同时进黄灯/全红
        (仿真时间共享, 切换期间未切换路口的绿灯自然延续并计入其相位时长),
        然后全体共同保持一段绿灯。actions: {tls_id: action}"""
        switching = [t for t in self.tls_ids if actions[t] != self.current_phase[t]]
        arrived = 0
        elapsed_trans = 0
        if switching:
            for tls in switching:
                self.traci.trafficlight.setRedYellowGreenState(
                    tls, yellow_state(self.green_states[tls][self.current_phase[tls]]))
            e, a = self._step_n(self.yellow_duration)
            elapsed_trans += e; arrived += a
            for tls in switching:
                self.traci.trafficlight.setRedYellowGreenState(
                    tls, all_red_state(self.green_states[tls][self.current_phase[tls]]))
            e, a = self._step_n(self.all_red_duration)
            elapsed_trans += e; arrived += a
            for tls in self.tls_ids:
                if tls in switching:
                    self.current_phase[tls] = actions[tls]
                    self.traci.trafficlight.setRedYellowGreenState(
                        tls, self.green_states[tls][actions[tls]])
                    self.phase_duration_sec[tls] = 0.0
                else:
                    self.phase_duration_sec[tls] += elapsed_trans

        remain = min((self.max_green_sec - self.phase_duration_sec[t]) for t in self.tls_ids)
        hold = int(min(self.hold_sec, max(remain, 1)))   # int: 干线过渡期+5s会使remain变浮点
        e, a = self._step_n(hold)
        arrived += a
        for tls in self.tls_ids:
            self.phase_duration_sec[tls] += e
        self._last_elapsed = max(elapsed_trans + e, 1)
        self._last_arrived = arrived
        return elapsed_trans + e

    def clamp_action(self, tls: str, proposed_action: int) -> int:
        """应用最小/最大绿灯约束(逐路口)"""
        if self.phase_duration_sec[tls] < self.min_green_sec:
            return self.current_phase[tls]
        if self.phase_duration_sec[tls] >= self.max_green_sec:
            return (self.current_phase[tls] + 1) % len(PHASE_NAMES)
        return proposed_action

    def compute_reward(self):
        """逐路口计算奖励(同一套tanh饱和公式), 返回 [(clipped, raw), ...]。
        吞吐量为全网共享, 按路口数均摊 —— 协同是团队目标。"""
        n = len(self.tls_ids)
        rewards = []
        for tls in self.tls_ids:
            total_waiting, veh_count = self._get_total_waiting_time_and_count(tls)
            avg_waiting = total_waiting / max(veh_count, 1)
            delta_avg_wait_per_sec = (avg_waiting - self._prev_avg_waiting[tls]) / self._last_elapsed
            self._prev_avg_waiting[tls] = avg_waiting
            self._last_mean_wait[tls] = avg_waiting

            throughput_per_sec = self._last_arrived / self._last_elapsed / n
            max_lane_wait = self._get_max_lane_waiting_time(tls)

            self._episode_wait_sum += avg_waiting / n
            self._episode_wait_max = max(self._episode_wait_max, max_lane_wait, avg_waiting)
            self._episode_delta_abs_sum += abs(delta_avg_wait_per_sec) / n
            self._episode_sample_count += 1

            wait_term = -self.reward_alpha * np.tanh(delta_avg_wait_per_sec / max(self._fixed_delta_scale, 1e-6))
            level_term = -self.reward_delay_coef * np.tanh(avg_waiting / max(self._fixed_wait_scale, 1e-6))
            fairness_term = -self.reward_fairness_coef * np.tanh(max_lane_wait / max(self._fixed_wait_scale, 1e-6))
            throughput_term = self.reward_gamma * throughput_per_sec

            raw = wait_term + level_term + throughput_term + fairness_term
            rewards.append((float(np.clip(raw, -self.reward_clip, self.reward_clip)), float(raw)))
        return rewards

    def _get_total_waiting_time_and_count(self, tls):
        lanes = set(self.traci.trafficlight.getControlledLanes(tls))
        total_wait = 0.0
        total_veh = 0
        for lane_id in lanes:
            total_wait += self.traci.lane.getWaitingTime(lane_id)
            total_veh += self.traci.lane.getLastStepVehicleNumber(lane_id)
        return total_wait, total_veh

    def _get_max_lane_waiting_time(self, tls):
        lanes = set(self.traci.trafficlight.getControlledLanes(tls))
        max_wait = 0.0
        for lane_id in lanes:
            w = self.traci.lane.getWaitingTime(lane_id)
            if w > max_wait:
                max_wait = w
        return max_wait

    def get_episode_scale_observations(self):
        """episode结束后调用, 返回本episode观测到的等待时间/变化率代表水平
        (多路口时为各路口的平均), 供调用方慢速校准下一轮的reward尺度。"""
        n = max(self._episode_sample_count, 1)
        observed_wait = self._episode_wait_sum / n
        observed_delta = self._episode_delta_abs_sum / n
        return observed_wait, observed_delta

    def is_finished(self) -> bool:
        if self.max_sim_time is not None and self.traci.simulation.getTime() >= self.max_sim_time:
            return True
        return self.traci.simulation.getMinExpectedNumber() <= 0
