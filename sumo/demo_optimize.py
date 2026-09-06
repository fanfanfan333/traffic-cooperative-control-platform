"""
问题二完整链路演示：Python决策 + SUMO评分
============================================
流程: 读数据 → Webster算初值 → SUMO跑分 → 爬山法优化 → 汇总

给算法队友的接口：score(greens) → 车均延误
"""
import os, sys, subprocess, xml.etree.ElementTree as ET, copy
import numpy as np, pandas as pd

# ===== 0. 环境 =====
SUMO = r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===== 1. 数据 =====
flow = pd.read_csv('../求解/预处理数据.csv')
sig  = pd.read_csv('../数据/团队数据V2  7-26/02_统一信号配时数据_V2.csv')

IID = 1           # 路口编号
PHASES = 4        # 信号相位对数（每个相位一对: 绿灯+黄灯+全红）
SAT = 1800 * 2    # 饱和流率
LOST = 4          # 每相位损失秒
dirs = [c for c in flow.columns if '进口' in c and c != '原始进口方向列表']

d = flow[(flow['路口编号']==IID) & (flow['时段分类']=='早高峰')]
h = d[dirs].mean() * 4
top = h[h>0].sort_values(ascending=False)

# ---------- Webster ----------
Y = sum(top.head(PHASES)) / SAT
C_web = max(60, min(180, int((1.5*PHASES*LOST + 5)/(1 - Y))))
eff = C_web - PHASES*LOST
crit = top.head(PHASES)
greens_web = [max(10, int(v/crit.sum()*eff)) for v in crit.values]

orig = sig[(sig['路口编号']==IID) & (sig['时段分类']=='早高峰')]
greens_orig = [int(r['绿灯时长(s)']) for _,r in orig.iterrows()][:PHASES]

print(f"路口{IID}")

# ===== 2. 核心函数：score(greens) → 延误 =====
# 策略：直接改 net.xml 里的相位时长，跑 SUMO，解析结果

# 先备份原始net
ORIG_NET = f'net/intersection_{IID}.net.xml'
with open(ORIG_NET, 'rb') as f:
    backup = f.read()

def score(greens):
    """
    输入: [绿灯1, 绿灯2, 绿灯3, 绿灯4] 单位秒
    输出: 车均等待时间(秒) — 越小越好
    """
    # 读取net.xml
    tree = ET.parse(ORIG_NET)
    root = tree.getroot()

    # 找tlLogic，改相位时长
    tl = root.find('.//tlLogic')
    if tl is None:
        return 999

    phases = tl.findall('phase')
    if len(phases) < PHASES * 2:
        return 999

    # 每个信号相位 = 绿灯phase + 黄灯/全红phase
    # 我们只改绿灯时长，间隔保持3s黄灯
    for i in range(PHASES):
        phases[i*2].set('duration', str(greens[i]))       # 绿灯
        phases[i*2+1].set('duration', str(LOST-1))       # 黄灯+全红(共LOST-1s,因为下个间隔还有1s)

    # 第二个间隔phase也在cycle里
    # 实际匹配原始net结构: 绿灯→黄灯(3s)→全红(1s) 共8个phase
    # 简化：只改4个绿灯phase
    for i in range(PHASES):
        idx = i * 2  # 第0,2,4,6个是绿灯phase
        if idx < len(phases):
            phases[idx].set('duration', str(greens[i]))

    # 写回
    tree.write(ORIG_NET, encoding='utf-8', xml_declaration=True)

    # 跑SUMO
    r = subprocess.run(
        [SUMO, '-c', f'intersection_{IID}.sumocfg',
         '--no-warnings', '--duration-log.statistics'],
        capture_output=True, text=True, timeout=60, cwd='.'
    )

    # 解析结果
    delay = 999
    trip_xml = f'results/tripinfo_{IID}.xml'
    if r.returncode == 0 and os.path.exists(trip_xml):
        try:
            tree2 = ET.parse(trip_xml)
            trips = tree2.getroot().findall('tripinfo')
            if trips:
                delay = sum(float(t.get('waitingTime', 0)) for t in trips) / len(trips)
        except ET.ParseError:
            pass  # XML可能被截断，跳过

    return delay

# ===== 3. 评估 =====
print("=" * 50)
print("SUMO 仿真打分")
print("=" * 50)

d_orig = score(greens_orig)
d_web = score(greens_web)

print(f"原始配时:    greens={greens_orig} → 延误={d_orig:.1f}s")
print(f"Webster:     greens={greens_web} → 延误={d_web:.1f}s")

# ===== 4. 爬山法 =====
print("\n" + "=" * 50)
print("爬山法迭代寻优")
print("=" * 50)

best = greens_web.copy()
best_d = d_web
print(f"起点: greens={best}, delay={best_d:.1f}s")

for gen in range(30):
    improved = False
    for pos in range(PHASES):
        for step in [+5, -5]:
            cand = best.copy()
            cand[pos] = max(10, cand[pos] + step)
            d = score(cand)
            if d < best_d:
                best, best_d = cand, d
                print(f"  gen{gen:2d}: 相位{pos+1}{'+' if step>0 else ''}{step}s "
                      f"→ greens={best}, delay={best_d:.1f}s")
                improved = True; break
        if improved: break
    if not improved:
        print(f"  gen{gen:2d}: 收敛!")
        break

# ===== 5. 汇总 =====
print("\n" + "=" * 50)
print("最终对比")
print("=" * 50)

# 恢复原始net
with open(ORIG_NET, 'wb') as f:
    f.write(backup)

final_orig = score(greens_orig)
final_web  = score(greens_web)
final_best = best_d  # already computed

print(f"{'方案':<12} {'周期':>6} {'绿灯':<20} {'延误':>6}")
print(f"{'原始':<12} {sum(greens_orig)+PHASES*LOST:>6} {str(greens_orig):<20} {final_orig:>6.1f}s")
print(f"{'Webster':<12} {C_web:>6} {str(greens_web):<20} {final_web:>6.1f}s")
print(f"{'爬山法最优':<12} {sum(best)+PHASES*LOST:>6} {str(best):<20} {final_best:>6.1f}s")
if final_orig > 0:
    print(f"\n改善率: {(final_orig-final_best)/final_orig*100:.1f}%")

pd.DataFrame([
    {'方案':'原始','绿灯':str(greens_orig),'周期':sum(greens_orig)+PHASES*LOST,'延误':round(final_orig,1)},
    {'方案':'Webster','绿灯':str(greens_web),'周期':C_web,'延误':round(final_web,1)},
    {'方案':'爬山法最优','绿灯':str(best),'周期':sum(best)+PHASES*LOST,'延误':round(final_best,1)},
]).to_csv('../求解/优化链路演示结果.csv', index=False, encoding='utf-8-sig')
print("已保存: 求解/优化链路演示结果.csv")
