import xml.etree.ElementTree as ET
import os

# 设置文件路径
images_dir = r"D:\BaiduNetdiskDownload\ILSVRC2012_img_val"
xmls_dir = r"E:\imageNet\ILSVRC2012_bbox_val_v3\val"
class_txt = r"E:\imageNet\ILSVRC2012_bbox_val_v3\class.txt"
label_txt = r"label1.txt"

# 加载类别到ID的映射
class_id_map = {}
try:
    # 尝试用GBk编码打开文件
    with open(class_txt, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割每一行，取前两列
            parts = line.strip().split()
            class_id_map[parts[1]] = parts[0]

except UnicodeDecodeError:
    print("尝试使用GBK编码打开文件失败，可能需要其他编码方式。")
except Exception as e:
    print(f"读取类别文件失败：{e}")
    class_id_map = {}

# 遍历目录中的所有图片文件
#
if class_id_map:
    label_count = 0
    with open(label_txt, 'w', encoding='gbk') as lf:  # 打开标签文件用于写入
        for img_file in os.listdir(images_dir):
            if img_file.endswith(".JPEG"):
                # 获取对应的XML文件名
                xml_file_name = img_file.split('.')[0] + ".xml"
                xml_file_path = os.path.join(xmls_dir, xml_file_name)

                # 检查对应的XML文件是否存在
                if not os.path.exists(xml_file_path):
                    print(f"对应的XML文件不存在：{xml_file_path}")
                    continue

                try:
                    # 解析XML文件获取类别信息
                    tree = ET.parse(xml_file_path)
                    root = tree.getroot()

                    # 查找第一个<name>标签
                    first_name = None
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        first_name = name
                        break  # 只取第一个name

                    if first_name and first_name in class_id_map:
                        class_id = class_id_map[first_name]
                        # 将结果写入标签文件
                        img_name = img_file.split('.')[0]
                        lf.write(f"{class_id}\n")
                        label_count += 1
                    elif not first_name:
                        print(f"XML文件中没有找到<name>标签：{xml_file_path}")
                    else:
                        print(f"类别名称未找到：{first_name}")
                except Exception as e:
                    print(f"处理XML文件失败：{e}, 文件名：{xml_file_name}")

    print(f"成功处理了 {label_count} 个标签")
else:
    print("类别映射字典为空，无法处理进一步的标签。")