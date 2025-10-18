import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def draw_heatmap2(gaze_points, display_size, alpha=0.5, savefilename=None, 
                imagefile=None, gaussianwh=200, gaussiansd=None):
    """
    生成基于高斯分布的黑白/彩色显著性热图
    
    参数：
    gaze_points: 注视点列表，格式为[(x1, y1, weight1), (x2, y2, weight2), ...]
    display_size: 屏幕分辨率 (width, height)
    alpha: 热图透明度（仅在有背景图时生效）
    savefilename: 保存路径
    imagefile: 背景图像路径
    gaussianwh: 高斯核大小（自动调整为奇数）
    gaussiansd: 高斯核标准差（None时自动计算为gaussianwh/4）
    """
    # 初始化参数
    display_width, display_height = display_size
    heatmap = np.zeros((display_height, display_width), dtype=np.float32)
    
    # 自动调整高斯核大小为奇数
    gaussianwh = int(gaussianwh)
    if gaussianwh % 2 == 0:
        gaussianwh += 1
    radius = gaussianwh // 2
    
    # 计算标准差
    if gaussiansd is None:
        gaussiansd = gaussianwh / 4  # 经验值
    
    # 生成高斯核
    kernel_1d = cv2.getGaussianKernel(gaussianwh, gaussiansd)
    kernel_2d = kernel_1d @ kernel_1d.T  # 二维高斯核
    kernel_2d /= kernel_2d.max()  # 归一化到[0,1]

    # 遍历所有注视点
    for point in gaze_points:
        x, y, weight = map(float, point)
        x = int(round(x))
        y = int(round(y))
        
        # 边界检查
        x = max(0, min(x, display_width-1))
        y = max(0, min(y, display_height-1))

        # 计算热力图叠加区域
        x_min, x_max = x - radius, x + radius + 1
        y_min, y_max = y - radius, y + radius + 1
        
        # 有效区域计算
        x_start = max(0, x_min)
        x_end = min(display_width, x_max)
        y_start = max(0, y_min)
        y_end = min(display_height, y_max)
        
        if x_start >= x_end or y_start >= y_end:
            continue  # 无有效区域可叠加

        # 计算高斯核截取范围
        kx_start = x_start - x_min
        kx_end = kx_start + (x_end - x_start)
        ky_start = y_start - y_min
        ky_end = ky_start + (y_end - y_start)
        
        # 截取有效高斯核并加权
        kernel_patch = kernel_2d[ky_start:ky_end, kx_start:kx_end] * weight
        
        # 叠加到热力图
        heatmap[y_start:y_end, x_start:x_end] += kernel_patch

    # 归一化处理
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # 可视化处理
    if imagefile:
        background = cv2.imread(imagefile)
        background = cv2.resize(background, (display_width, display_height))
        heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(background, 1-alpha, heatmap_colored, alpha, 0)
    else:
        blended = (heatmap * 255).astype(np.uint8)

    # 保存结果
    if savefilename:
        cv2.imwrite(savefilename, blended)
    
    return blended

# 设置输入路径和输出路径
input_path = 'D:/1PHD/methods/3method_trans/gazedata240828/NC/test'
output_dir = os.path.join(input_path, 'combined_heatmaps')
os.makedirs(output_dir, exist_ok=True)

# 定义需要处理的5个saliency任务
tasks = ['saliency1', 'saliency2', 'saliency3', 'saliency4', 'saliency5']

# 处理每个saliency任务
for task in tasks:
    combined_data = []
    
    # 遍历所有用户（1-50号文件夹）
    for user_id in tqdm(range(1, 51), desc=f'Processing {task}'):
        user_folder = os.path.join(input_path, str(user_id))
        task_file = os.path.join(user_folder, f"{task}.txt")
        
        # 如果文件不存在则跳过
        if not os.path.exists(task_file):
            continue
            
        # 读取并处理文件内容
        with open(task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 解析坐标数据（假设格式：左眼X 左眼Y 右眼X 右眼Y）
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                # 添加左右眼数据（转换为整数并添加权重）
                _, lx, ly, rx, ry = map(float, parts)
                combined_data.append((int(lx), int(ly), 1))  # 左眼数据
                combined_data.append((int(rx), int(ry), 1))  # 右眼数据

    # 生成热力图
    if combined_data:
        output_path = os.path.join(output_dir, f"{task}_combined_heatmap.png")
        draw_heatmap2(combined_data, 
                    (1920, 1080), 
                    alpha=0.5,
                    savefilename=output_path,
                    imagefile=None,
                    gaussianwh=200,
                    gaussiansd=None)