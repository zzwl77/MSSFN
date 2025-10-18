import pandas as pd
from gazegraph import *
import csv
import os
from tqdm import tqdm
# 设置保存文件的根目录
data_root = 'D:\\1PHD\\methods\\3method_trans\\gazedata240828'

# 读取包含所有CSV文件地址的文件
csv_files = []
with open('D:\\1PHD\\methods\\3method_trans\\gazedata240828\\prepare_list.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        parent_dir_name = os.path.basename(row[0])
        csv_file = os.path.join(row[0], parent_dir_name + '.csv')
        csv_files.append(row[0])

# 定义任务范围和对应的背景图像
task_ranges = {
    "presaccade1": (3562, 3947),
    "presaccade2": (3947, 4379),
    "antisaccade1": (4824, 5258),
    "antisaccade2": (5258, 5639),
    "sensitivity1": (7515, 7814),
    "sensitivity2": (7913, 8212),
    "sensitivity3": (8314, 8613),
    "saliency1": (13930, 14229),
    "saliency2": (14290, 14589),
    "saliency3": (14650, 14949),
    "saliency4": (15010, 15309),
    "saliency5": (15370, 15669),
    "color1": (15730, 16029),
    "color2": (16090, 16389),
    "color3": (16450, 16749),
    "color4": (16810, 17109)
}

total_tasks = len(csv_files) * len(task_ranges)
# 循环处理每个任务
with tqdm(total=total_tasks, desc='Processing') as pbar:
    for task, task_range in task_ranges.items():
        for csv_file in csv_files:
            task_data = []
            output_directory = os.path.join("D:\\1PHD\\methods\\3method_trans\\heatshow", csv_file)
            csv_file = os.path.join(data_root, csv_file, csv_file.split('\\')[-1] + '.csv')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            # 为左右数据分别设置不同的文件名后缀
            output_name_l = os.path.join(output_directory, f"{task}_l.pickle")
            output_name_r = os.path.join(output_directory, f"{task}_r.pickle")

            with open(csv_file, 'r') as file:
                reader = csv.reader(file)

                for row in reader:
                    frame_number = int(float(row[0]))

                    if task_range[0] <= frame_number <= task_range[1]:
                        task_data.append(row)

            if task_data:
                gaze_data_l = [(float(q[0]), float(q[1]), float(q[2])) for q in task_data]
                gaze_data_r = [(float(q[0]), float(q[3]), float(q[4])) for q in task_data]

                # 分别绘制左右数据的热图
                G_left = draw_graph2(gaze_data_l, k=10, output_name=output_name_l, max_nodes=50)
                # G_right  = draw_graph3(gaze_data_r, k=10, output_name=output_name_r, max_nodes=50)

                plot_graph(G_left, f"{task}_l")
                # plot_graph(G_right, f"{task}_r")
                pbar.update(1)