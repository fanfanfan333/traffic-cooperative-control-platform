# -*- coding: utf-8 -*-
"""
SUMO建模脚本 v3：统一简单路网 + 自动edge映射 + 需求/信号/配置
"""
import pandas as pd
import numpy as np
import os, sys, glob, subprocess, xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

SUMO_HOME = r'C:\Program Files (x86)\Eclipse\Sumo'
NETCONVERT = os.path.join(SUMO_HOME, 'bin', 'netconvert.exe')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

SUMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(SUMO_DIR)
DATA = os.path.join(PROJ, '数据', '团队数据V2  7-26')
NET_DIR = os.path.join(SUMO_DIR, 'net')
ROUTE_DIR = os.path.join(SUMO_DIR, 'routes')
RESULT_DIR = os.path.join(SUMO_DIR, 'results')
for d in [NET_DIR, ROUTE_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
inter_df = pd.read_excel(os.path.join(DATA, '03_路口基础信息表_V2.xlsx'))
flow_df = pd.read_csv(os.path.join(PROJ, '求解', '预处理数据.csv'))
signal_df = pd.read_csv(os.path.join(DATA, '02_统一信号配时数据_V2.csv'))

# ============================================================
# Part 1: 为所有20个路口生成标准简单路网
# ============================================================
print("=" * 60)
print("Part 1: 生成统一简单路网")
print("=" * 60)

def gen_net_4way(iid):
    """标准十字路口（4进口）"""
    nod = '''<?xml version="1.0" encoding="UTF-8"?>
<nodes>
<node id="J" x="0" y="0" type="traffic_light"/>
<node id="E0" x="-300" y="0"/><node id="W0" x="300" y="0"/>
<node id="N0" x="0" y="300"/><node id="S0" x="0" y="-300"/>
</nodes>'''
    edg = '''<?xml version="1.0" encoding="UTF-8"?>
<edges>
<edge id="E_in" from="E0" to="J" numLanes="3" speed="13.89"/>
<edge id="E_out" from="J" to="E0" numLanes="2" speed="13.89"/>
<edge id="W_in" from="W0" to="J" numLanes="3" speed="13.89"/>
<edge id="W_out" from="J" to="W0" numLanes="2" speed="13.89"/>
<edge id="N_in" from="N0" to="J" numLanes="3" speed="13.89"/>
<edge id="N_out" from="J" to="N0" numLanes="2" speed="13.89"/>
<edge id="S_in" from="S0" to="J" numLanes="3" speed="13.89"/>
<edge id="S_out" from="J" to="S0" numLanes="2" speed="13.89"/>
</edges>'''
    return nod, edg, {'东':['E_in'],'西':['W_in'],'北':['N_in'],'南':['S_in']}, \
           {'东':['E_out'],'西':['W_out'],'北':['N_out'],'南':['S_out']}

def gen_net_3way_T(iid):
    """T字路口（3进口: 东、西、南）"""
    nod = '''<?xml version="1.0" encoding="UTF-8"?>
<nodes>
<node id="J" x="0" y="0" type="traffic_light"/>
<node id="E0" x="-300" y="0"/><node id="W0" x="300" y="0"/>
<node id="S0" x="0" y="-300"/><node id="N0" x="0" y="300"/>
</nodes>'''
    edg = '''<?xml version="1.0" encoding="UTF-8"?>
<edges>
<edge id="E_in" from="E0" to="J" numLanes="3" speed="13.89"/>
<edge id="E_out" from="J" to="E0" numLanes="2" speed="13.89"/>
<edge id="W_in" from="W0" to="J" numLanes="3" speed="13.89"/>
<edge id="W_out" from="J" to="W0" numLanes="2" speed="13.89"/>
<edge id="S_in" from="S0" to="J" numLanes="3" speed="13.89"/>
<edge id="S_out" from="J" to="S0" numLanes="2" speed="13.89"/>
<edge id="N_out" from="J" to="N0" numLanes="2" speed="13.89"/>
</edges>'''
    return nod, edg, {'东':['E_in'],'西':['W_in'],'南':['S_in']}, \
           {'东':['E_out'],'西':['W_out'],'南':['S_out']}

def gen_net_5way(iid):
    """五进口路口"""
    return gen_net_4way(iid)  # simplified to 4-way

# 映射每路口的类型
nets = {}
for _, row in inter_df.iterrows():
    iid = int(row['路口编号'])
    itype = str(row['路口类型'])
    imports = int(row['进口数'])

    if 'T字' in itype or 'T' in itype or '三进口' in itype or imports == 3:
        nod, edg, app_in, app_out = gen_net_3way_T(iid)
    elif '五进口' in itype or imports == 5:
        nod, edg, app_in, app_out = gen_net_5way(iid)
    else:
        nod, edg, app_in, app_out = gen_net_4way(iid)

    nf = os.path.join(NET_DIR, f'tmp_{iid}.nod.xml')
    ef = os.path.join(NET_DIR, f'tmp_{iid}.edg.xml')
    with open(nf, 'w', encoding='utf-8') as f: f.write(nod)
    with open(ef, 'w', encoding='utf-8') as f: f.write(edg)

    net_out = os.path.join(NET_DIR, f'intersection_{iid}.net.xml')
    r = subprocess.run([NETCONVERT, '--node-files', nf, '--edge-files', ef,
                        '-o', net_out, '--tls.guess', '--tls.default-type', 'static',
                        '--no-turnarounds', 'true'], capture_output=True, text=True)
    os.remove(nf); os.remove(ef)

    if r.returncode == 0:
        nets[iid] = {'in': app_in, 'out': app_out, 'tl_ids': ['J']}
        print(f"  intersection_{iid} ({itype}, {imports}进口): OK")
    else:
        print(f"  intersection_{iid}: FAILED")

# ============================================================
# Part 2: 生成路线文件 (data→SUMO demand)
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 生成SUMO需求路线")
print("=" * 60)

dir_cols = [c for c in flow_df.columns if '进口' in c and c != '原始进口方向列表']

# 交通方向 → cardinal方向映射
data_to_card = {
    '东进口左转': ('东', '北'), '东进口直行': ('东', '西'), '东进口右转': ('东', '南'),
    '西进口左转': ('西', '南'), '西进口直行': ('西', '东'), '西进口右转': ('西', '北'),
    '北进口左转': ('北', '西'), '北进口直行': ('北', '南'), '北进口右转': ('北', '东'),
    '南进口左转': ('南', '东'), '南进口直行': ('南', '北'), '南进口右转': ('南', '西'),
}

for iid in range(1, 21):
    if iid not in nets:
        continue

    app_in = nets[iid]['in']
    app_out = nets[iid]['out']
    inter_flow = flow_df[flow_df['路口编号'] == iid]
    hourly_means = inter_flow[dir_cols].mean() * 4  # pcu/h

    routes, flows = [], []
    rid, fid, total_vph = 0, 0, 0

    for dcol in dir_cols:
        fv = hourly_means.get(dcol, 0)
        if pd.isna(fv) or fv <= 0 or dcol not in data_to_card:
            continue

        fc, tc = data_to_card[dcol]
        if fc not in app_in or tc not in app_out:
            continue

        from_e = app_in[fc][0]
        to_e = app_out[tc][0]
        vph = max(1, int(round(float(fv))))
        total_vph += vph

        routes.append(f'  <route id="r_{rid}" edges="{from_e} {to_e}"/>')
        flows.append(f'  <flow id="f_{fid}" type="car" route="r_{rid}" begin="0" end="3600" vehsPerHour="{vph}"/>')
        rid += 1; fid += 1

    if rid == 0:
        print(f"  intersection_{iid}: WARNING - 0 routes")
        continue

    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
           '  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
           '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5"',
           '         length="5.0" minGap="2.5" maxSpeed="13.89"/>']
    xml.extend(routes); xml.extend(flows); xml.append('</routes>')

    with open(os.path.join(ROUTE_DIR, f'intersection_{iid}.rou.xml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml))

    if iid <= 3 or iid in [11, 12, 20]:
        print(f"  intersection_{iid}: {rid} routes, {total_vph} veh/h")

# ============================================================
# Part 3: 信号配时additional文件
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 生成信号配时")
print("=" * 60)

for iid in range(1, 21):
    if iid not in nets:
        continue

    inter_signal = signal_df[signal_df['路口编号'] == iid]
    if len(inter_signal) == 0:
        continue

    pkey = {'早高峰': 'morning', '平峰': 'noon', '晚高峰': 'evening'}

    for period in ['早高峰', '平峰', '晚高峰']:
        ps = inter_signal[inter_signal['时段分类'] == period]
        if len(ps) == 0:
            continue

        xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<additional>']
        for tl_id in nets[iid]['tl_ids']:
            xml.append(f'  <tlLogic id="{tl_id}" type="static" programID="{tl_id}_{pkey[period]}" offset="0">')
            for _, row in ps.iterrows():
                g = int(row['绿灯时长(s)'])
                y = int(row['黄灯时长(s)']) if not pd.isna(row['黄灯时长(s)']) else 3
                r = int(row['全红时长(s)']) if not pd.isna(row['全红时长(s)']) else 0
                dur = g + y + r
                s = 'G' * 12  # simplified full-green state
                xml.append(f'    <phase duration="{dur}" state="{s}"/>')
            xml.append('  </tlLogic>')
        xml.append('</additional>')

        with open(os.path.join(ROUTE_DIR, f'intersection_{iid}_{pkey[period]}.add.xml'),
                  'w', encoding='utf-8') as f:
            f.write('\n'.join(xml))

    if iid <= 3:
        print(f"  intersection_{iid}: 3 periods OK")

# ============================================================
# Part 4: SUMO配置文件 + 路口类型元数据
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 生成SUMO配置")
print("=" * 60)

intersection_meta = {}  # for later use by problem scripts

for iid in range(1, 21):
    if iid not in nets:
        continue

    sumocfg = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="net/intersection_{iid}.net.xml"/>
        <route-files value="routes/intersection_{iid}.rou.xml"/>
    </input>
    <time><begin value="0"/><end value="3600"/></time>
    <output>
        <tripinfo-output value="results/tripinfo_{iid}.xml"/>
    </output>
    <processing><time-to-teleport value="-1"/></processing>
</configuration>'''

    cfg_path = os.path.join(SUMO_DIR, f'intersection_{iid}.sumocfg')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(sumocfg)

    # 存储元数据
    info = inter_df[inter_df['路口编号'] == iid]
    intersection_meta[iid] = {
        'type': info['路口类型'].values[0] if len(info) > 0 else '未知',
        'approaches': int(info['进口数'].values[0]) if len(info) > 0 else 4,
    }

total_cfg = len(glob.glob(os.path.join(SUMO_DIR, 'intersection_*.sumocfg')))
print(f"生成 {total_cfg} 个 .sumocfg")

# ============================================================
# 测试运行 intersection_1
# ============================================================
print("\n" + "=" * 60)
print("测试运行 SUMO 仿真")
print("=" * 60)

sumo_exe = os.path.join(SUMO_HOME, 'bin', 'sumo.exe')
cfg1 = os.path.join(SUMO_DIR, 'intersection_1.sumocfg')
r = subprocess.run([sumo_exe, '-c', cfg1, '--no-warnings', '--duration-log.statistics'],
                   capture_output=True, text=True, cwd=SUMO_DIR)
if r.returncode == 0:
    # 解析结果
    trip_path = os.path.join(RESULT_DIR, 'tripinfo_1.xml')
    if os.path.exists(trip_path):
        tree = ET.parse(trip_path)
        trips = tree.getroot().findall('tripinfo')
        if trips:
            durations = [float(t.get('duration', 0)) for t in trips]
            wait_times = [float(t.get('waitingTime', 0)) for t in trips]
            print(f"  intersection_1: {len(trips)} trips completed")
            print(f"    avg duration: {np.mean(durations):.1f}s, avg wait: {np.mean(wait_times):.1f}s")
        else:
            print(f"  intersection_1: No trips completed")
else:
    print(f"  intersection_1: Failed - {r.stderr[:300]}")

print("\nSUMO模型构建完成！")
