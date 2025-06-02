# 计算准确率
correct = 0
total = 0

# 读取真实类别和预测类别文件
with open(r"ground_truth_label.txt", 'r') as f_label, \
     open(r"mid6_13_cut_predict_2pad.txt", 'r') as f_predict:
    # 逐行读取并比较
    for line_label, line_predict in zip(f_label, f_predict):
        # 解析真实类别
        img_name_label, true_class = line_label.strip().split(': ')
        # 解析预测类别
        img_name_predict, predicted_class = line_predict.strip().split(': ')
        # 确保比较的是同一张图片
        if img_name_label != img_name_predict:
            print(f"图片名称不匹配：{img_name_label} vs {img_name_predict}")
            continue
        # 判断预测是否正确
        if true_class == predicted_class:
            correct += 1
        total += 1

# 计算准确率
accuracy = correct / total if total != 0 else 0
print(f"模型准确率: {accuracy:.2%} ({correct}/{total})")