"""
最简TraCI演示：Python控制SUMO一步步跑
跑之前先确保 sumo_app/routes/intersection_1.rou.xml 存在
"""
import os, sys

# 第1步：告诉Python SUMO装在哪
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

# 第2步：启动SUMO（-c 后面是配置文件路径）
sumo_exe = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
traci.start([sumo_exe, "-c", "intersection_1.sumocfg", "--no-warnings"])

# 第3步：一步步跑，每步1秒
wait_times = []
for step in range(100):                            # 只跑100秒做演示
    traci.simulationStep()                         # ← 推进1秒

    # 拿这1秒的数据
    veh_ids = traci.vehicle.getIDList()            # 所有在路上跑的车
    if veh_ids:
        waits = [traci.vehicle.getWaitingTime(v)   # 每辆车等了多少秒
                 for v in veh_ids]
        avg_wait = sum(waits) / len(waits)
        wait_times.append(avg_wait)

    # 打印当前状态（每10秒输出一次）
    if step % 10 == 0:
        print(f"第{step:4d}秒 | 路上{len(veh_ids):3d}辆车 | 均等待{sum(waits)/len(waits):.1f}s"
              if veh_ids else f"第{step:4d}秒 | 路上  0辆车")

# 第4步：收工
traci.close()
print(f"\n完成！共模拟100秒")
print(f"平均车均等待: {sum(wait_times)/len(wait_times):.1f}s")
