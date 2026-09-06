# 交通预测流 — YOLO8 + DeepSORT + DQN (+ SUMO 仿真训练)

## 系统架构

```
视频帧                                    或  SUMO 仿真 (sumo_pipeline.py)
  │                                            │  TraCI 仿真真值
  ▼                                            ▼
┌─────────────────────────────────────┐
│         YOLOv8 目标检测器            │
│  · Ultralytics YOLOv8n/s/m/l/x     │
│  · 交通类别过滤 (car/bus/truck/...)  │
│  · 可选 ROI 裁切                     │
└──────────────┬──────────────────────┘
               │ [bbox, score, class_id]
               ▼
┌─────────────────────────────────────┐
│         DeepSORT 多目标追踪          │
│  ┌────────────┐  ┌───────────────┐  │
│  │  卡尔曼滤波 │  │  ReID 特征提取 │  │
│  │  运动预测  │  │  外观嵌入     │  │
│  └──────┬─────┘  └──────┬────────┘  │
│         └───────┬────────┘          │
│          ┌──────▼──────┐            │
│          │  匈牙利匹配  │            │
│          │ 两阶段策略   │            │
│          └─────────────┘            │
└──────────────┬──────────────────────┘
               │ [track_id, bbox, velocity]
               ▼
┌─────────────────────────────────────┐
│       交通状态构建器                  │
│  · 车道密度/队列/速度                │
│  · 当前相位 one-hot (4维)            │
│  · 相位持续时间归一化 (1维)           │
│  → 状态向量 s ∈ ℝ²⁹ (视频) / ℝ⁴¹ (SUMO)│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Dueling Double DQN 智能体       │
│  ┌──────────────────────────────┐   │
│  │       共享特征层              │   │
│  │    FC(→128) → ReLU           │   │
│  │    FC(128→64) → ReLU         │   │
│  └────────┬─────────────┬───────┘   │
│     ┌─────▼────┐  ┌─────▼────┐      │
│     │ 价值流 V │  │ 优势流 A  │      │
│     │  FC→1   │  │  FC→4    │      │
│     └─────┬───┘  └─────┬───────┘      │
│           └──────┬───────┘          │
│              Q = V + A - mean(A)   │
│                                     │
│  + Prioritized Experience Replay    │
│  + 3-step 经验回放 (n-step 可调)     │
│  + Huber 损失 (抗离群TD误差)         │
│  + 学习率 warmup + 余弦退火          │
│  + 目标网络 Polyak 软更新 (τ=0.005)  │
│  + ε-greedy 探索衰减                │
└──────────────┬──────────────────────┘
               │ 动作: 下一信号相位
               ▼
         交通信号控制器
```

### 动作相位语义：与真实路口结构对齐（重要）

`sumo_state_builder.PHASE_NAMES` 定义 4 个动作对应的信号放行方式：

| 动作 | 相位 | 放行方式 |
|------|------|---------|
| 0 | 南北向直行左转 | 直行+右转 绿灯(G)，左转**让行绿**(g) |
| 1 | 南北向左转 | 仅左转保护绿灯(G) |
| 2 | 东西向直行左转 | 直行+右转 绿灯(G)，左转让行绿(g) |
| 3 | 东西向左转 | 仅左转保护绿灯(G) |

这套语义与典型固定配时的"混合放行 + 左转保护"结构一致，修复了两个历史 bug：
1. **右转全程红灯**：旧版 4 个相位名里都没有"右转"，右转 link 被全程压红，
   右转车流无限堆积把路口拖垮（实测延误 110s+，完成车辆只有固定配时的 60%）；
2. **直左拆分导致绿灯利用率低**：旧版直行相位不含左转，智能体只能机械轮转
   4 相位，绿灯利用率远低于混合放行（实测路口1延误 20s vs 固定配时 13.5s）。

改为混合相位后，原始固定配时的相位序列成为智能体策略空间的子集，
实测智能体学会了"跳过不需要的左转保护相位"这一关键技巧。

**注意：相位语义变更后，旧版环境训练的 .pt 模型的动作含义已改变，需要用新环境重新训练。**

## 实测结果（20路口数据集，2026-09）

固定 3600s 仿真，DQN 为评估模式(ε=0) 3 种子平均；基线为同种子静态配时。

**路口1**（需求 1853 辆/h，需求均衡）

| 方案 | 车均延误 | 完成车辆 | 对比 |
|------|---------|---------|------|
| **DQN (本项目)** | **4.75s** | 1829 (98.7%) | 基准 |
| 原始固定配时 | 13.59s | 1828 | DQN 好 65% |
| Webster | 22.19s | 1816 | DQN 好 79% |
| 爬山优化 | 18.41s (seed0) | - | DQN 好 74% |

**路口18**（需求 2837 辆/h，东西向主导，近饱和）

| 方案 | 车均延误 | 完成车辆 | 对比 |
|------|---------|---------|------|
| **DQN (本项目)** | **4.05s** | 2798 (98.6%) | 基准 |
| 原始固定配时 | 81.17s | 2185 (77%) | DQN 好 95% |
| Webster | 57.72s | 2600 (92%) | DQN 好 93% |
| 爬山优化 | 24.48s (seed0) | - | DQN 好 83% |

学到的策略：路口1 在南北/东西混合相位间 50/50 轮转、完全跳过左转保护相位；
路口18 东西向占 100%（73% 混合 + 27% 左转保护），南北向需求小仅维持最小绿灯。

消融证据: `dqn_i1_old_phases.csv` / `dqn_i18_old_phases.csv`（旧相位语义下路口1卡在20s，印证相位对齐的必要性）。

### 统一模型：一个模型通吃 20 单路口 + 3 干线

用 `--sumocfg all` 轮换训练（每轮随机切换场景，reward 尺度按路口分别校准），
最终模型 `dqn_unified.pt`（61 维状态，累计 295 轮）。评估（ε=0）：

| 指标 | 原始固定配时 | 统一 DQN |
|------|------------|---------|
| 20路口平均车均延误 | 19.33s | **4.58s (降76%)** |
| 单路口改善幅度 | - | 32%~96% 全部为正 |
| 干线改善幅度 | - | 91%~94%（见下节） |
| 吞吐量 | 部分路口仅77% | 全部 98.5%+ |

亮点：路口19 延误 47.65s→2.04s（降96%）、路口18 79.85s→4.17s（降95%）；
统一模型在单路口与干线上均无专项模型的性能损失，即"一个模型部署到所有场景"。
训练曲线: `dqn_unified_curves.png`；逐轮日志: `dqn_unified.csv`；
评估记录: `dqn_eval_unified.csv`（230轮版）/ `dqn_eval_unified_after.csv`（295轮终版）。

## 干线绿波协同（多路口控制）

状态空间统一为 **61 维**（单路口 41 维 + 4 方向邻居槽位 × 5 维邻居相位上下文），
单路口邻居槽位恒为 0 —— **一个模型同时覆盖 20 个单路口与 3 条干线**。
环境层 `SumoTrafficEnv` 为多信号灯版：各路口同时决策、同时黄灯切换，
奖励按路口分别计算（吞吐项均摊），共享策略逐路口存取经验；
邻居相位上下文让路口"看到"上游刚放行的车队（约 36s 后抵达），提前腾空绿灯。

进口方向识别采用**几何法**（按边起点节点相对路口的方位），彻底替代按边名
前缀映射 —— 干线走廊边 `link_X` 及任意命名均可正确归并（旧命名法曾在干线
走廊上复现"方向永红"类 bug）。

### 干线实测结果（3600s，tripinfo 口径，3种子均值，最终模型=295轮统一模型）

| 干线 | 基线(内置42/42同步程序) | DQN 协同管控 | 改善 | 吞吐 |
|------|----------------------|-------------|------|------|
| 干线一(4路口) | 103.79s | **7.57s** | **93%** | 满载* |
| 干线二(5路口) | 137.39s | **7.66s** | **94%** | 满载* |
| 干线三(2路口) | 40.51s | **3.34s** | **92%** | 满载* |

20个单路口(同模型同口径): 平均 4.58s vs 原始配时 19.33s。

*吞吐为3600s视界内完成数，剩余车辆为截止前进场未走完2km走廊，非容量损失。
短窗口(600s)对比: DQN 通过量 **+40%**、等待 **-96%**、停车 **-80%**、油耗 **-24%**
(`reports_arterial/report.md` + 时空轨迹图: 绿色斜带=车队一路绿波直通)。

注意: 数据集自带的 `coord` 协调程序存在缺陷 —— 12字符状态串套在11连接路口上,
走廊方向永无绿灯, 0/800辆完成(已在采集器中用 `--program 0` 切回内置程序做基线,
该数据缺陷的发现与诊断本身也是平台验证能力的证明)。

## 功能二 · 数据采集与动态可视化（对标赛题"数据采集与可视化模块"）

采用 **"端-边-云"三层架构**（呼应车路云一体化主题）：

```
端(TraCI逐秒采样)        边(本地增量聚合)              云(展示与决策支持)
┌────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 车辆轨迹(x,y,v, │ → │ 进口道排队几何归并     │ → │ ECharts实时仪表盘     │
│ 瞬时油耗)       │   │ O(1)累计KPI           │   │ (dashboard_server.py)│
│ 车道排队/相位    │   │ 停车事件在线检测       │   │ 时空轨迹图/KPI报告    │
└────────────────┘   │ 油耗积分(标定mg/s→ml/s)│   │ (visualize_results.py)│
                     └──────────────────────┘   └──────────────────────┘
```

创新点：多源数据统一时空基准融合（轨迹+排队+相位+**油耗**，TraCI油耗返回值
经密度标定校准为标准ml/s）；时空轨迹图**自动检测并叠加停车事件**（红叉标注，
可直观看出车堵在哪、堵多久）；**双方案同屏实时对比**（同种子下固定配时vs
DQN，KPI卡片实时显示改善百分比）；一键生成**标准化评估报告**（Markdown，
覆盖赛题指标集：效率/排队/通行能力/能耗/停车）。

### 使用

```bash
# 1. 采集(同场景同种子各跑一次, 数据直接可比)
python data_collector.py --sumocfg intersection_1.sumocfg --control fixed --label fixed
python data_collector.py --sumocfg intersection_1.sumocfg --control dqn \
    --model dqn_unified.pt --label dqn

# 2a. 实时动态仪表盘(浏览器打开 http://127.0.0.1:8080, 演示录屏建议全屏)
python dashboard_server.py

# 2b. 离线标准化评估报告(reports/ 下: 时空轨迹图+对比曲线+KPI汇总+report.md)
python visualize_results.py --baseline fixed --proposal dqn

# 训练/评估管线也可挂采集器: 加 --collect-data 即可
python sumo_pipeline.py --eval --episodes 1 --model dqn_unified.pt --collect-data
```

实测对比（路口1，600s仿真，seed=0，详见 `reports/report.md`）：
DQN 相对固定配时 —— 平均等待 **-81%**、平均排队 **-67%**、人均停车 **-41%**、
油耗 **-13%**、平均车速 **+18%**、通过量 +1.4%，六个维度全部改善。

## 核心模块

| 文件 | 功能 |
|------|------|
| `yolo_detector.py` | YOLOv8 封装，支持 ROI / 批推理 |
| `deepsort_tracker.py` | 卡尔曼滤波 + 匈牙利匹配 + ReID |
| `traffic_dqn.py` | 状态空间 + 奖励函数 + Dueling DDQN + PER + n-step + 学习率调度 |
| `pipeline.py` | 视频主管线：YOLO检测 → DeepSORT追踪 → DQN控制 |
| `sumo_pipeline.py` | SUMO 仿真训练/评估主管线（单路口/多路口轮换，状态为仿真真值） |
| `sumo_traffic_env.py` | SUMO/TraCI 环境封装：信号控制 + tanh饱和奖励 |
| `sumo_state_builder.py` | SUMO 41维状态构建（4方向×3转向×3特征+相位） |
| `delay_score.py` | 比赛口径延误打分 + Webster/爬山基线对比 |
| `data_collector.py` | 功能二·端边层数据采集（轨迹/排队/相位/油耗，停车事件在线检测） |
| `dashboard_server.py` | 功能二·云端实时仪表盘（ECharts，双方案同屏对比） |
| `visualize_results.py` | 功能二·离线评估报告（时空轨迹图/KPI汇总/report.md） |
| `plot_training.py` | 训练曲线绘图（延误+loss，多路口着色） |
| `dqn_selftest.py` | 训练核心自测脚本（不依赖SUMO，验证正确性与学习能力） |

## 安装

```bash
pip install torch ultralytics opencv-python scipy numpy sumolib
```

另需安装 [SUMO](https://sumo.dlr.de/docs/Installing.html)（默认路径 `D:\Sumo`，或设置 `SUMO_HOME` 环境变量）。

## 快速使用

### SUMO 仿真训练（推荐）

```bash
# 默认跑路口1, 单轮
python sumo_pipeline.py

# 多轮训练(推荐): 连续跑50轮, 每轮独立随机车流种子
python sumo_pipeline.py --episodes 50 --model dqn_traffic_sumo.pt

# 通用模型: 一个模型吃下全部20个路口(每轮轮换一个路口,
# reward尺度校准按路口分别维护), 训练/评估通用
python sumo_pipeline.py --sumocfg all --episodes 230 --model dqn_unified.pt

# 也可以逗号分隔指定部分路口
python sumo_pipeline.py --sumocfg "intersection_1.sumocfg,intersection_18.sumocfg" --episodes 50

# 评估模式(不训练, 加载已有模型, eps=0); 配合 --sumocfg all 可批量测评全部路口
python sumo_pipeline.py --eval --episodes 23 --sumocfg all --model dqn_unified.pt

# 训练曲线图(延误+loss, 多路口自动按路口着色)
python plot_training.py dqn_unified.csv

# 换路口(只需给 cfg 名, net 会从 sumocfg 里自动读)
python sumo_pipeline.py --sumocfg intersection_2.sumocfg
```

### 训练曲线可视化

```bash
python plot_training.py dqn_unified.csv --quick        # 单路口日志也可画: 延误+loss双图
python plot_training.py a.csv b.csv --compare         # 多文件同图对比
python plot_training.py dqn_unified.csv                # 多路口: 按路口着色散点
```

### 视频训练

```bash
# 训练模式（视频文件）
python pipeline.py --source traffic.mp4 --yolo yolov8n.pt --display

# 评估模式（不训练，加载已有模型）
python pipeline.py --source traffic.mp4 --model dqn_traffic.pt --eval --display

# 使用 GPU 加速
python pipeline.py --source 0 --device cuda --yolo yolov8s.pt
```

## 训练核心设计（近期优化重点）

### 学习率策略：3e-3 + warmup + 余弦退火

学习率基准值从 `1e-3` 提高到 **`3e-3`**，直接加快收敛；同时用两层保障确保
"高学习率"不发散：

1. **线性 warmup**（`--lr-warmup`，默认500步）：冷启动阶段 Q 值估计还很离谱，
   先把学习率从接近 0 线性升到 3e-3，避免开局就被大步长踢飞；
2. **余弦退火**（`--lr-total-steps`，默认50000步）：warmup 结束后按余弦曲线
   平滑退火到基准值的 10%（`--lr-min-scale`，即 3e-4），前期大学习率快速逼近
   好策略，后期小学习率精细收敛、不破坏已学到的策略。

调度进度随模型一起保存/恢复；加载旧模型时以本次传入的 `--lr` 为基准重新缩放。

### 高学习率的稳定器

| 机制 | 说明 |
|------|------|
| Huber 损失 | 大误差区间梯度有上界，个别离群样本不再能把训练带飞（`--loss mse` 可切回旧版对比） |
| 目标网络软更新 | `θ_target ← (1-τ)θ_target + τθ_online`，τ=0.005，训练目标平滑漂移（`--target-update-tau`，设0退回每200步硬拷贝） |
| 大批量 | batch 64→128，梯度噪声更小，支撑更高学习率 |
| 梯度裁剪 | grad-norm clip 10.0 |
| loss 熔断 | 检测到 NaN/Inf 直接跳过本次更新，防止网络被永久污染 |

### 3-step 经验回放（n-step）

一次决策约 10 秒绿灯，1-step TD 需要很多轮才能把奖励信号传回早期决策。
改为默认 **3-step 折扣回报**（`--n-step 3`），信用分配速度快约3倍；
done 时正确截断，episode 结束自动 flush 尾巴经验。设 `--n-step 1` 退化为旧版。

### PER bug 修复

旧版 `push()` 对新样本优先级多做了一次 `**alpha`（树内存的是已做过 alpha 次幂的值），
导致新经验的采样概率被系统性压低、几乎采不到，已在 `traffic_dqn.py` 修正。

### 探索策略

epsilon 每训练步衰减乘数从 0.995 放缓到 **0.997**（`--epsilon-decay`），
多留 2~3 轮充分探索，避免过早收敛到"偏科"策略（实测中出现过某方向绿灯占比
从 ~25% 涨到 ~50%、另一方向被压到不足 10% 的问题，配合奖励中的公平性惩罚项缓解）。

## 状态空间设计

**视频管线（29维）**

```
[lane_0_density, lane_0_queue, lane_0_speed,   # 车道 N_L1
 ...                                            # × 8 车道
 phase_NS_straight, phase_NS_left,             # 当前相位 one-hot
 phase_EW_straight, phase_EW_left,
 phase_duration_normalized]                    # 相位持续时间
```

**SUMO 管线（41维）**：4进口方向 × 3转向 × (密度/排队/速度) + 4相位 one-hot + 相位时长

## 奖励函数（SUMO 管线）

```
r = -α·tanh(ΔWait/s_d) - β·tanh(Wait/s_w) + γ·Throughput - φ·tanh(MaxLaneWait/s_w)
```

各项经 tanh 平滑饱和，同一套默认参数可同时适配轻度/严重拥堵路口；
含公平性惩罚（`--reward-fairness-coef`）防止长期饿死某方向，
等待尺度参考值跨 episode 自动校准并持久化（`*.calib.json`）。

## 奖励函数（视频管线）

```
r = -α·ΔWait - β·ΔQueue + γ·Throughput - δ·Emergency

α=1.0  等待时间变化权重
β=0.5  队列长度变化权重
γ=2.0  通过车辆数奖励
δ=5.0  紧急车辆阻塞惩罚
```

## 自测

不依赖 SUMO，几分钟内验证训练核心的正确性与学习能力（PER不变量、n-step回报
手算比对、学习率调度、save/load、合成MDP上新旧超参收敛速度对比）：

```bash
python dqn_selftest.py          # 全部测试
python dqn_selftest.py --quick  # 跳过学习能力对比
```

## 替换 ReID 模型

将 ONNX 格式的 ReID 模型路径传入：
```bash
python pipeline.py --reid osnet_x0_25.onnx ...
```
推荐模型：OSNet-x0.25（轻量）、FastReID-ResNet50（高精度）
