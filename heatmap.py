import os
import csv
from PIL import Image
import pandas as pd
from utils.preprocess.gazeheatplot import draw_heatmap
import csv
import os
from tqdm import tqdm

import os
import csv
from PIL import Image
import pandas as pd
from utils.preprocess.gazeheatplot import draw_heatmap
from tqdm import tqdm

class HeatmapGenerator:
    def __init__(self, root_path):
        self.root_path = root_path
        self.task_ranges = {
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
        self.background_images = {task: None for task in self.task_ranges}

    def combine_csv(self):
        file_names = ['presaccade1.txt', 'presaccade2.txt', 'antisaccade1.txt', 'antisaccade2.txt', 'seneitivity1.txt', 'seneitivity2.txt', 'seneitivity3.txt', 'saliency1.txt', 'saliency2.txt', 'saliency3.txt', 'saliency4.txt', 'saliency5.txt', 'color1.txt', 'color2.txt', 'color3.txt', 'color4.txt']
        data = []
        for file_name in file_names:
            with open(os.path.join(self.root_path, file_name), 'r') as f:
                for line in f:
                    delimiter = ',' if ',' in line else None
                    row_data = line.strip().split(delimiter)
                    data.append(row_data[:5])
        with open(os.path.join(self.root_path, self.root_path.split('\\')[-1] + '.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def gen_heatmap(self):
        total_tasks = len(self.task_ranges)
        with tqdm(total=total_tasks, desc='Processing') as pbar:
            for task, task_range in self.task_ranges.items():
                task_data = []
                csv_file = os.path.join(self.root_path, self.root_path.split('\\')[-1] + '.csv')
                output_directory = os.path.join(self.root_path, "heatshow")
                os.makedirs(output_directory, exist_ok=True)
                output_name_l = os.path.join(output_directory, f"{task}_l.png")
                output_name_r = os.path.join(output_directory, f"{task}_r.png")
                background_image = self.background_images[task]

                with open(csv_file, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        frame_number = int(float(row[0]))
                        if task_range[0] <= frame_number <= task_range[1]:
                            task_data.append(row)

                if task_data:
                    gaze_data_l = [(int(float(q[1])), int(float(q[2])), 1) for q in task_data]
                    gaze_data_r = [(int(float(q[3])), int(float(q[4])), 1) for q in task_data]
                    draw_heatmap(gaze_data_l, (1920, 1080), alpha=0.5, savefilename=output_name_l, imagefile=background_image, gaussianwh=200, gaussiansd=None)
                    draw_heatmap(gaze_data_r, (1920, 1080), alpha=0.5, savefilename=output_name_r, imagefile=background_image, gaussianwh=200, gaussiansd=None)
                pbar.update(1)
    
    def convert_and_save_images(self):
        for filename in os.listdir(self.root_path):
            file_path = os.path.join(self.root_path, filename)
            if os.path.isdir(file_path):
                # 如果是子文件夹，递归处理该子文件夹
                self.convert_and_save_images(file_path)
            elif filename.endswith(".png"):
                # 如果是图像文件
                # 打开RGBA图像
                rgba_image = Image.open(file_path)
                # 将RGBA图像转换为RGB格式
                rgb_image = rgba_image.convert("RGB")
                # 覆盖原图像并保存为PNG格式
                rgb_image.save(file_path)
                print(f"Converted and saved: {file_path}")

def main():
    root_path = 'D:\\1PHD\\methods\\3method_trans\\gazedata240828'
    heatmap_gen = HeatmapGenerator(root_path)
    heatmap_gen.combine_csv()
    heatmap_gen.gen_heatmap()
    heatmap_gen.convert_and_save_images()

if __name__ == "__main__":

    main()
