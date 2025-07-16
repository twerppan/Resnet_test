import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# 设备算力配置
device_flops = {
    'rpi5_1': 320e9,
    'rpi5_2': 320e9,
    'rpi5_3': 320e9,
    'jetson1': 472e9,
    'jetson2': 472e9,
    'jetson3': 472e9,
    'pc_cpu': 500e9,
}


def generate_load_table(num_periods, devices):
    """
    生成设备负载变化表
    :param num_periods: 时间段数量（每个时间段5分钟）
    :param devices: 设备列表
    :return: 包含时间戳和负载数据的DataFrame
    """
    # 时间序列生成 (5分钟间隔)
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(num_periods)]

    # 创建DataFrame
    df = pd.DataFrame(index=timestamps)

    # 为每个设备生成负载模式
    for device in devices:
        # 基础负载模式 (基于设备类型)
        if 'rpi5' in device:
            # Raspberry Pi: 周期性负载变化，白天高负载
            base_load = 0.1 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, num_periods))
            # 添加工作日模式
            weekday_effect = np.array([0.1 if ts.weekday() < 5 else -0.1 for ts in timestamps])
            base_load += weekday_effect
        elif 'jetson' in device:
            # Jetson: 更稳定的负载，偶尔有峰值
            base_load = 0.2 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, num_periods))
            # 添加随机峰值
            peaks = np.random.choice([0, 0.3], size=num_periods, p=[0.95, 0.05])
            base_load += peaks
        else:
            # PC CPU: 相对稳定的高负载
            base_load = 0.4 + 0.1 * np.sin(np.linspace(0, np.pi, num_periods))

        # 添加设备特定的随机波动
        noise = np.random.normal(0, 0.05, num_periods)
        load = np.clip(base_load + noise, 0.05, 0.95)

        # 确保负载在合理范围内
        df[device] = np.round(load, 3)

    return df


def generate_bandwidth_table(devices, num_periods=100):
    """
    生成异构设备间的带宽表（含时间维度）
    :param devices: 设备列表
    :param num_periods: 时间段数量
    :return: 包含所有时间段带宽数据的长格式DataFrame
    """
    num_devices = len(devices)

    # 初始化三维数据立方体 (设备A × 设备B × 时间段)
    bandwidth_cube = np.zeros((num_devices, num_devices, num_periods))

    # 设备分组
    rpi_devices = [d for d in devices if 'rpi5' in d]
    jetson_devices = [d for d in devices if 'jetson' in d]
    pc_devices = [d for d in devices if 'pc_cpu' in d]

    # 生成每个时间段的带宽
    for period in range(num_periods):
        bandwidth = np.zeros((num_devices, num_devices))

        # 同组设备间带宽 (高带宽)
        for i, dev1 in enumerate(devices):
            for j, dev2 in enumerate(devices):
                if i == j:
                    continue  # 设备自身带宽为0

                # 添加周期性波动 (白天带宽高，夜晚带宽低)
                time_factor = 0.8 + 0.2 * np.sin(2 * np.pi * period / num_periods)

                if dev1 in rpi_devices and dev2 in rpi_devices:
                    # Raspberry Pi之间: 基础带宽 + 随机波动 + 时间因子
                    base_bw = random.uniform(100, 150)
                    bw_variation = random.uniform(0.9, 1.1)
                    bandwidth[i, j] = base_bw * bw_variation * time_factor
                elif dev1 in jetson_devices and dev2 in jetson_devices:
                    # Jetson之间: 基础带宽 + 随机波动 + 时间因子
                    base_bw = random.uniform(200, 300)
                    bw_variation = random.uniform(0.9, 1.1)
                    bandwidth[i, j] = base_bw * bw_variation * time_factor
                elif dev1 in pc_devices and dev2 in pc_devices:
                    continue  # 只有一个PC，保持为0
                elif (dev1 in rpi_devices and dev2 in jetson_devices) or (
                        dev1 in jetson_devices and dev2 in rpi_devices):
                    # RPi与Jetson之间
                    base_bw = random.uniform(50, 80)
                    bw_variation = random.uniform(0.9, 1.1)
                    bandwidth[i, j] = base_bw * bw_variation * time_factor
                elif (dev1 in rpi_devices and dev2 in pc_devices) or (dev1 in pc_devices and dev2 in rpi_devices):
                    # RPi与PC之间
                    base_bw = random.uniform(80, 120)
                    bw_variation = random.uniform(0.9, 1.1)
                    bandwidth[i, j] = base_bw * bw_variation * time_factor
                elif (dev1 in jetson_devices and dev2 in pc_devices) or (dev1 in pc_devices and dev2 in jetson_devices):
                    # Jetson与PC之间
                    base_bw = random.uniform(150, 250)
                    bw_variation = random.uniform(0.9, 1.1)
                    bandwidth[i, j] = base_bw * bw_variation * time_factor

        # 确保对称性
        for i in range(num_devices):
            for j in range(i + 1, num_devices):
                avg_bw = (bandwidth[i, j] + bandwidth[j, i]) / 2
                bandwidth[i, j] = avg_bw
                bandwidth[j, i] = avg_bw

        bandwidth_cube[:, :, period] = bandwidth

    # 创建长格式DataFrame包含所有时间段
    bandwidth_list = []
    for period in range(num_periods):
        for i, dev1 in enumerate(devices):
            for j, dev2 in enumerate(devices):
                if i != j:  # 排除设备自身的连接
                    bandwidth_list.append({
                        'period': period + 1,
                        'from_device': dev1,
                        'to_device': dev2,
                        'bandwidth_mbps': bandwidth_cube[i, j, period]
                    })

    return pd.DataFrame(bandwidth_list).round(1)


# 生成100个时间段的负载表
load_table = generate_load_table(100, list(device_flops.keys()))

# 生成包含100个时间段的带宽表
bandwidth_table = generate_bandwidth_table(list(device_flops.keys()))

# 保存到CSV文件
load_table.to_csv('device_load_table_100_periods.csv')
bandwidth_table.to_csv('device_bandwidth_table_100_periods.csv', index=False)

# 打印部分结果
print(f"设备负载变化表 (100个时间段, 前5个时间点):")
print(load_table.head())

print("\n带宽表 (前10条记录):")
print(bandwidth_table.head(10))

# 可视化负载变化
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
for device in device_flops:
    plt.plot(load_table.index, load_table[device], label=device)

plt.title('Device Load Over 100 Periods (5-min intervals)')
plt.xlabel('Time')
plt.ylabel('Load (0-1)')
plt.legend(title='Devices', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('device_load_plot_100_periods.png', dpi=300)
plt.show()

# 可视化带宽变化（示例：rpi5_1到jetson1的带宽变化）
plt.figure(figsize=(12, 6))
# 筛选特定设备对的带宽数据
rpi_to_jetson = bandwidth_table[
    (bandwidth_table['from_device'] == 'rpi5_1') &
    (bandwidth_table['to_device'] == 'jetson1')
    ].sort_values('period')

plt.plot(load_table.index[:100], rpi_to_jetson['bandwidth_mbps'], 'g-', linewidth=2)

plt.title('Bandwidth Variation: rpi5_1 to jetson1 (100 Periods)')
plt.xlabel('Time')
plt.ylabel('Bandwidth (MB/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bandwidth_variation_example.png', dpi=300)
plt.show()

# 可视化特定时间段的带宽关系（示例：第1个时间段）
period1_bandwidth = bandwidth_table[bandwidth_table['period'] == 1].pivot(
    index='from_device', columns='to_device', values='bandwidth_mbps'
).fillna(0)

plt.figure(figsize=(10, 8))
plt.imshow(period1_bandwidth.values, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Bandwidth (MB/s)')
plt.xticks(range(len(period1_bandwidth.columns)), period1_bandwidth.columns, rotation=45)
plt.yticks(range(len(period1_bandwidth.index)), period1_bandwidth.index)
plt.title('Device-to-Device Bandwidth (Period 1)')
plt.tight_layout()
plt.savefig('bandwidth_matrix_period1.png', dpi=300)
plt.show()