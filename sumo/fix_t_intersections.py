# -*- coding: utf-8 -*-
"""修复T字路口路网——根据实际的缺失进口方向生成正确的edges"""
import pandas as pd
import os, subprocess, xml.etree.ElementTree as ET

SUMO_HOME = r'C:\Program Files (x86)\Eclipse\Sumo'
NETCONVERT = os.path.join(SUMO_HOME, 'bin', 'netconvert.exe')

SUMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(SUMO_DIR)
DATA = os.path.join(PROJ, '数据', '团队数据V2  7-26')
NET_DIR = os.path.join(SUMO_DIR, 'net')
ROUTE_DIR = os.path.join(SUMO_DIR, 'routes')
os.makedirs(NET_DIR, exist_ok=True)
os.makedirs(ROUTE_DIR, exist_ok=True)

inter_df = pd.read_excel(os.path.join(DATA, '03_路口基础信息表_V2.xlsx'))

# Each T-intersection needs different approaches
# The config tells us which approach is MISSING
# We need to create edges for the remaining 3 approaches

def gen_net_from_approaches(iid, approaches):
    """
    approaches: list of cardinal directions that EXIST
    e.g. ['西', '北', '南'] means E is missing
    """
    nodes = ['<?xml version="1.0" encoding="UTF-8"?>', '<nodes>']
    edges = ['<?xml version="1.0" encoding="UTF-8"?>', '<edges>']

    # Central junction
    nodes.append('<node id="J" x="0" y="0" type="traffic_light"/>')

    card_positions = {
        '东': (-300, 0),    # entry from left
        '西': (300, 0),     # entry from right
        '北': (0, 300),     # entry from top
        '南': (0, -300),    # entry from bottom
    }

    # Track which outgoing edges we need
    outgoing = {'东': 'W_out', '西': 'E_out', '北': 'S_out', '南': 'N_out'}
    incoming = {'东': 'E_in', '西': 'W_in', '北': 'N_in', '南': 'S_in'}

    for card in approaches:
        x, y = card_positions[card]
        nodes.append(f'<node id="{card}_entry" x="{x}" y="{y}"/>')

        in_edge_id = incoming[card]
        out_edge_id = outgoing[card]
        edges.append(f'<edge id="{in_edge_id}" from="{card}_entry" to="J" numLanes="3" speed="13.89"/>')
        edges.append(f'<edge id="{out_edge_id}" from="J" to="{card}_entry" numLanes="2" speed="13.89"/>')

    nodes.append('</nodes>')
    edges.append('</edges>')

    nf = os.path.join(NET_DIR, f'tmp_fix_{iid}.nod.xml')
    ef = os.path.join(NET_DIR, f'tmp_fix_{iid}.edg.xml')
    with open(nf, 'w', encoding='utf-8') as f: f.write('\n'.join(nodes))
    with open(ef, 'w', encoding='utf-8') as f: f.write('\n'.join(edges))

    net_out = os.path.join(NET_DIR, f'intersection_{iid}.net.xml')
    r = subprocess.run([NETCONVERT, '--node-files', nf, '--edge-files', ef,
                        '-o', net_out, '--tls.guess', '--tls.default-type', 'static',
                        '--no-turnarounds', 'true'], capture_output=True, text=True)
    os.remove(nf); os.remove(ef)

    if r.returncode == 0:
        return True, approaches
    return False, []

# Map intersection special notes to existing approaches
def get_existing_approaches(special_note):
    """Parse the 特殊进口朝向 field to determine which approaches exist."""
    all_cards = {'东', '西', '北', '南'}

    if pd.isna(special_note) or special_note == '无':
        return list(all_cards)

    note = str(special_note)

    # Direct missing statements
    if '东进口不存在' in note or '无东进口' in note:
        return ['西', '北', '南']
    if '西进口不存在' in note or '无西进口' in note:
        return ['东', '北', '南']
    if '北进口不存在' in note or '无北进口' in note:
        return ['东', '西', '南']
    if '南进口不存在' in note or '无南进口' in note:
        return ['东', '西', '北']

    # Special approach specifications
    if '东北' in note and '西南' in note:
        return ['东', '西', '北', '南']  # 4-way, use standard
    if '东北' in note:
        # Has a 东北 approach in addition or instead
        remaining = list(all_cards - {'北'}) if '北' not in note else list(all_cards)
        if len(remaining) <= 2:
            return remaining
        return remaining

    # For "东北-西-南" style notes
    if '-' in note:
        parts = note.replace(' ', '').split('-')
        result = []
        card_map = {'东': '东', '西': '西', '北': '北', '南': '南',
                    '东北': '东', '西北': '西', '东南': '东', '西南': '西'}
        for p in parts:
            if p in card_map:
                c = card_map[p]
                if c not in result:
                    result.append(c)
        return result if result else list(all_cards)

    return list(all_cards)


# Fix all T-intersections
fixes = {}
for _, row in inter_df.iterrows():
    iid = int(row['路口编号'])
    itype = str(row['路口类型'])
    imports = int(row['进口数'])
    special = str(row['特殊进口朝向'])

    # Only fix T-intersections and 3-进口 ones
    if 'T' not in itype and '三进口' not in itype and imports != 3:
        continue

    approaches = get_existing_approaches(special)
    print(f"路口{iid} ({itype}, 缺失说明: {special}): 实际进口 → {approaches}")

    if len(approaches) == 3:
        ok, apps = gen_net_from_approaches(iid, approaches)
        if ok:
            fixes[iid] = apps
            print(f"  → 修复成功！edges: E_in/E_out={('东' in apps)}, W_in/W_out={('西' in apps)}, N_in/N_out={('北' in apps)}, S_in/S_out={('南' in apps)}")
    else:
        # 4-way, just keep existing net
        print(f"  → 已有4进口，保持不变")

print(f"\n修复完成: {len(fixes)} 个路口")
for iid, apps in fixes.items():
    print(f"  路口{iid}: {apps}")
