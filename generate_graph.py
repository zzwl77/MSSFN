import pandas as pd
from utils.preprocess.gazegraph import *
import csv
import os
from tqdm import tqdm
from matplotlib.collections import LineCollection

# 定义任务范围和对应的背景图像
def visualize_scientific_graph(node_features, time_features, edge_index, output):
    """
    Visualize gaze graph with academic paper style
    
    Parameters:
    node_features (Tensor): Normalized node coordinates [N, 2]
    time_features (Tensor): Temporal features [N]
    edge_index (Tensor): Graph connectivity [2, E]
    output_path (str): Output file path (PDF recommended)
    """
    # Convert tensor to numpy
    nodes = node_features.numpy()
    times = time_features.numpy()
    edges = edge_index.numpy().T
    
    # Create figure with appropriate size
    plt.figure(figsize=(4, 3), dpi=300)
    ax = plt.gca()

    # Configure plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # Plot edges with alpha blending
    edge_lines = []
    for (src, dst) in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        edge_lines.append([(x1, y1), (x2, y2)])
    
    lc = LineCollection(edge_lines, 
                       colors=(0.7, 0.7, 0.7, 0.3),  # RGBA (light gray with 30% alpha)
                       linewidths=0.8,
                       zorder=1)
    ax.add_collection(lc)

    # Plot nodes with temporal coloring
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1],
                        c=times,
                        cmap='viridis',
                        s=40,
                        edgecolor='w',
                        linewidth=0.5,
                        zorder=2)

    # Configure axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel('Normalized X Coordinate',fontsize=12)
    ax.set_ylabel('Normalized Y Coordinate',fontsize=12)
    ax.set_aspect('equal')

    # Remove top/right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Save publication-ready figure
    plt.savefig(output, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()

def visualize_scientific_graph2(node_features, time_features, edge_index, output):
    """
    Visualize gaze graph with academic paper style
    
    Parameters:
    node_features (Tensor): Normalized node coordinates [N, 2]
    time_features (Tensor): Temporal features [N]
    edge_index (Tensor): Graph connectivity [2, E]
    output_path (str): Output file path (PDF recommended)
    """
    # Convert tensor to numpy
    nodes = node_features.numpy()
    times = time_features.numpy()
    edges = edge_index.numpy().T
    
    # Create figure with 9:16 aspect ratio (portrait orientation)
    plt.figure(figsize=(4, 2.25), dpi=300)  # 4.5:8 = 9:16 ratio
    ax = plt.gca()

    # Configure plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
    })

    # Plot edges with alpha blending
    edge_lines = []
    for (src, dst) in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        edge_lines.append([(x1, y1), (x2, y2)])
    
    lc = LineCollection(edge_lines, 
                       colors=(0.7, 0.7, 0.7, 0.3),  # RGBA (light gray with 30% alpha)
                       linewidths=0.8,
                       zorder=1)
    ax.add_collection(lc)

    # Plot nodes with temporal coloring
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1],
                        c=times,
                        cmap='viridis',
                        s=40,
                        edgecolor='w',
                        linewidth=0.5,
                        zorder=2)

    # Configure axes and remove them
    # ax.set_xlim(0, 8)
    # ax.set_ylim(4.5, 0)  # Maintain top-left origin
    # ax.set_aspect('equal')
    # ax.axis('off')  # Remove axes completely
    # 添加边框
    ax.set_xlim(0, 8)
    ax.set_ylim(4.5, 0)  # Maintain top-left origin (0,0) is top-left
    ax.set_aspect('equal')
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    # 移除刻度线和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    # Save publication-ready figure
    plt.savefig(output, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def visualize_scientific_graph3(node_features, time_features, edge_index, output, background_image=None):
    """
    Visualize gaze graph with academic paper style
    
    Parameters:
    node_features (Tensor): Normalized node coordinates [N, 2]
    time_features (Tensor): Temporal features [N]
    edge_index (Tensor): Graph connectivity [2, E]
    output (str): Output file path (PDF recommended)
    background_image (str): Optional path to background image
    """
    # Convert tensor to numpy
    nodes = node_features.numpy() if hasattr(node_features, 'numpy') else node_features
    times = time_features.numpy() if hasattr(time_features, 'numpy') else time_features
    edges = edge_index.numpy().T if hasattr(edge_index, 'numpy') else edge_index.T
    
    # Create figure with 9:16 aspect ratio (portrait orientation)
    plt.figure(figsize=(4, 2.25), dpi=300)  # 4.5:8 = 9:16 ratio
    ax = plt.gca()

    # Configure plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
    })

    # Load and display background image if provided
    if background_image:
        try:
            bg_img = plt.imread(background_image)
            # Display image matching coordinate system [xmin, xmax, ymin, ymax]
            ax.imshow(bg_img, 
                      extent=[0, 8, 4.5, 0],  # Matches set_xlim/set_ylim ranges
                      aspect='auto', 
                      alpha=1, 
                      zorder=0)  # Ensure background stays behind everything
        except Exception as e:
            print(f"Error loading background image: {e}")

    # Plot edges with alpha blending
    edge_lines = []
    for (src, dst) in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        edge_lines.append([(x1, y1), (x2, y2)])
    
    lc = LineCollection(edge_lines, 
                       colors=(0.2, 0.2, 0.2, 0.7),  # RGBA (light gray with 30% alpha)
                       linewidths=0.8,
                       zorder=1)  # Above background
    ax.add_collection(lc)

    # Plot nodes with temporal coloring
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1],
                        c=times,
                        cmap='inferno',
                        s=50,
                        edgecolor='w',
                        linewidth=1.0,
                        zorder=2)  # Above edges and background

    # Configure axes
    ax.set_xlim(0, 8)
    ax.set_ylim(4.5, 0)  # Maintain top-left origin (0,0) is top-left
    ax.set_aspect('equal')
    
    # # Set border style
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    #     spine.set_color('black')
    #     spine.set_linewidth(1.5)
    
    # # Remove ticks and labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.tick_params(axis='both', which='both', length=0)
    # 完全移除所有边框和坐标轴
    ax.set_axis_off()  # 这行代码移除所有轴线和标签
    
    # 完全移除所有空白边距
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Save publication-ready figure
    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def main():
    data_root = 'D:\\1PHD\\methods\\3method_trans\\gazedata240828'

    # 读取包含所有CSV文件地址的文件
    csv_files = []
    with open('D:\\1PHD\\methods\\3method_trans\\gazedata240828\\prepare_list.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            parent_dir_name = os.path.basename(row[0])
            csv_file = os.path.join(row[0], parent_dir_name + '.csv')
            csv_files.append(row[0])
            
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
    background_images = {
    task_name: os.path.join(data_root, "taskmaps", f"{i+1}.png") for i, task_name in enumerate(task_ranges)
    }
    total_tasks = len(csv_files) * len(task_ranges)

    with tqdm(total=total_tasks, desc='Processing') as pbar:
        for task, task_range in task_ranges.items():
            for csv_file in csv_files:
                task_data = []
                output_directory = os.path.join("D:\\1PHD\\methods\\3method_trans\\graphshow", csv_file)
                csv_file = os.path.join(data_root, csv_file, csv_file.split('\\')[-1] + '.csv')
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                # 为左右数据分别设置不同的文件名后缀
                output_name_l = os.path.join(output_directory, f"{task}_l.pickle")
                background_image = background_images[task]

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
                    node_features, time_features, edge_features, edge_index = draw_graph4(gaze_data_l, k=10, output_name=output_name_l)
                    output_path_l = os.path.join(output_directory, f"{task}_l.png")
                    visualize_scientific_graph3(node_features, time_features, edge_index, output_path_l, background_image)

                    node_features, time_features, edge_features, edge_index = draw_graph4(gaze_data_r, k=10, output_name=output_name_l)
                    output_path_r = os.path.join(output_directory, f"{task}_r.png")
                    visualize_scientific_graph3(node_features, time_features, edge_index, output_path_r, background_image)
                    
                    pbar.update(1)


if __name__ == '__main__':
    main()