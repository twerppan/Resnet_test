import numpy as np
import random
from itertools import permutations


class NeuralNetwork:
    def __init__(self):
        self.layers = [
            {"name": "x", "shape": "(1, 3, 224, 224)", "size": 150528, "flops": 0, "is_cut": True},
            {"name": "conv1", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 43.3521, "is_cut": True},
            {"name": "bn1", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 1.60563, "is_cut": True},
            {"name": "relu1", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 0.802816, "is_cut": True},
            {"name": "conv2", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 231.211, "is_cut": True},
            {"name": "bn2", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 1.60563, "is_cut": True},
            {"name": "relu2", "shape": "(1, 16, 224, 224)", "size": 802816, "flops": 0.802816, "is_cut": True},
            {"name": "pool", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.200704, "is_cut": False},
            {"name": "conv3", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 57.8028, "is_cut": False},
            {"name": "bn3", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.401408, "is_cut": False},
            {"name": "relu3", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.200704, "is_cut": False},
            {"name": "conv4", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 57.8028, "is_cut": False},
            {"name": "bn4", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.401408, "is_cut": False},
            {"name": "add", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.200704, "is_cut": True},
            {"name": "relu", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0.200704, "is_cut": False},
            {"name": "br1_conv", "shape": "(1, 8, 112, 112)", "size": 100352, "flops": 3.21126, "is_cut": False},
            {"name": "br1_bn", "shape": "(1, 8, 112, 112)", "size": 100352, "flops": 0.200704, "is_cut": False},
            {"name": "br2_conv", "shape": "(1, 8, 112, 112)", "size": 100352, "flops": 3.21126, "is_cut": False},
            {"name": "br2_bn", "shape": "(1, 8, 112, 112)", "size": 100352, "flops": 0.200704, "is_cut": False},
            {"name": "cat", "shape": "(1, 16, 112, 112)", "size": 200704, "flops": 0, "is_cut": True},
            {"name": "global_pool", "shape": "(1, 16, 1, 1)", "size": 16, "flops": 1.6e-05, "is_cut": True},
            {"name": "flatten", "shape": "(1, 16)", "size": 16, "flops": 0, "is_cut": True},
        ]

    def get_segment_info(self, start_idx, end_idx):
        total_flops = sum(layer["flops"] for layer in self.layers[start_idx:end_idx + 1])
        output_size = self.layers[end_idx]["size"]
        return total_flops, output_size


class DeviceManager:
    def __init__(self, devices):
        self.devices = devices
        self.device_combinations = list(permutations(devices, 3)) #生成所有三设备的排列组合

    def validate_allocation(self, flops_list, device_order):#验证FLOPs分配是否满足设备剩余能力
        return all(flop <= dev["remaining_flops"] for flop, dev in zip(flops_list, device_order))


class MultiArmedBandit:
    def __init__(self, num_arms):#各臂被选择的次数
        self.num_arms = num_arms
        self.estimated_rewards = np.full(num_arms, np.nan)  # 各臂的估计奖励值
        self.num_pulls = np.zeros(num_arms)#各臂被选择的次数

    def select_arm(self, epsilon):
        # 优先选择未探索的臂
        unexplored = np.where(self.num_pulls == 0)[0]
        if unexplored.size > 0:
            return random.choice(unexplored)

        # 处理全NaN情况
        if np.all(np.isnan(self.estimated_rewards)):
            return random.randint(0, self.num_arms - 1)

        # epsilon-greedy策略
        if random.random() < epsilon:
            return random.randint(0, self.num_arms - 1)
        return np.nanargmax(self.estimated_rewards)

    def update(self, chosen_arm, reward):#增量平均法更新奖励估计
        self.num_pulls[chosen_arm] += 1
        n = self.num_pulls[chosen_arm]
        current = self.estimated_rewards[chosen_arm]

        # 处理首次更新时的NaN
        if np.isnan(current):
            self.estimated_rewards[chosen_arm] = reward
        else:
            self.estimated_rewards[chosen_arm] = current + (reward - current) / n


def calculate_reward(device_flops, segment_flops, output_size):
    # 处理边界条件
    if device_flops <= 0:
        return -np.inf  # 设备无计算能力

    if segment_flops > device_flops:
        return -100  # 超限惩罚

    # 添加数值稳定性保护
    utilization = segment_flops / (device_flops + 1e-8)
    transmission = output_size / (1024 ** 2)

    return -0.7 * utilization - 0.3 * transmission

def generate_valid_arms(model, device_manager):
    cut_points = [i for i, layer in enumerate(model.layers) if layer["is_cut"]]
    valid_arms = []#收集所有可切割点

    for i in range(len(cut_points)):#遍历所有两切割点组合（cp1, cp2
        for j in range(i + 1, len(cut_points)):
            cp1, cp2 = cut_points[i], cut_points[j]
            if cp1 >= cp2:
                continue

            # 计算各段计算量
            seg1_flops = sum(l["flops"] for l in model.layers[:cp1 + 1])
            seg2_flops = sum(l["flops"] for l in model.layers[cp1 + 1:cp2 + 1])
            seg3_flops = sum(l["flops"] for l in model.layers[cp2 + 1:])
            flops_list = [seg1_flops, seg2_flops, seg3_flops]

            # 生成有效设备分配
            for device_order in device_manager.device_combinations:
                if device_manager.validate_allocation(flops_list, device_order):
                    valid_arms.append({
                        "cut_points": (cp1, cp2),
                        "devices": device_order,
                        "segments_flops": flops_list
                    })

    return valid_arms


def main():
    model = NeuralNetwork()
    devices = [
        {"id": 1, "remaining_flops": 100},
        {"id": 2, "remaining_flops": 400},
        {"id": 3, "remaining_flops": 200}
    ]

    device_manager = DeviceManager(devices)
    valid_arms = generate_valid_arms(model, device_manager)

    min_device_flops = min(d["remaining_flops"] for d in devices)
    if min_device_flops <= 0:
        raise ValueError("设备计算能力必须大于0")

    # 添加有效方案检查
    if not valid_arms:
        print("没有找到有效的切割方案！")
        print("可能原因：")
        print("1. 设备计算能力不足（总需求：{:.1f}，当前设备总和：{:.1f}）".format(
            sum(l["flops"] for l in model.layers),
            sum(d["remaining_flops"] for d in devices)))
        print("2. 可用切割点不足（当前有效切割点：{})".format(len(valid_arms)))

    bandit = MultiArmedBandit(len(valid_arms))
    epsilon = 0.1
    iterations = 5000

    for iter in range(iterations):
        arm_idx = bandit.select_arm(epsilon)
        arm = valid_arms[arm_idx]
        cp1, cp2 = arm["cut_points"]
        devices = arm["devices"]
        flops_list = arm["segments_flops"]

        # 获取各段输出大小
        _, out1 = model.get_segment_info(0, cp1)
        _, out2 = model.get_segment_info(cp1 + 1, cp2)
        _, out3 = model.get_segment_info(cp2 + 1, len(model.layers) - 1)
        print(flops_list[0],flops_list[1],flops_list[2])
        # 计算各段奖励
        r1 = calculate_reward(devices[0]["remaining_flops"], flops_list[0], out1)
        r2 = calculate_reward(devices[1]["remaining_flops"], flops_list[1], out2)
        r3 = calculate_reward(devices[2]["remaining_flops"], flops_list[2], out3)
        print(r1,r2,r3)

        total_reward = (r1 + r2 + r3) / 3
        print(total_reward)
        bandit.update(arm_idx, total_reward)

    # 输出最终结果
    best_idx = np.nanargmax(bandit.estimated_rewards)
    best_arm = valid_arms[best_idx]
    print("\n最佳切割方案：")
    print(f"切割点：{best_arm['cut_points']}")
    print(f"设备分配：{[d['id'] for d in best_arm['devices']]}")
    print(f"预测奖励：{bandit.estimated_rewards[best_idx]:.2f}")

if __name__ == "__main__":
    main()