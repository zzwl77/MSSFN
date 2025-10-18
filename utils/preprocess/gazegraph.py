import pickle
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
def read_data(file_path):
    return pd.read_csv(file_path)

# 计算两个点之间的欧几里得距离
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 计算两点间的方向角
def calculate_orientation(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

# 构建单眼的k-hop STGG图，包含方向信息
def draw_graph(gaze_data, k, output_name):
    G = nx.DiGraph()
    positions = {}

    # 提取 x 和 y 的最大值和最小值
    # x_coords = [coord[0] for index, coord in gaze_data]
    # y_coords = [coord[1] for index, coord in gaze_data]
    
    # x_min, x_max = min(x_coords), max(x_coords)
    # y_min, y_max = min(y_coords), max(y_coords)

    # 对 x 和 y 进行归一化
    gaze_data_normalized = []
    for index, (_, (x, y)) in enumerate(gaze_data):
        # x_norm = (x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
        # y_norm = (y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
        x_norm = x / 1920
        y_norm = y / 1080
        gaze_data_normalized.append((index, (x_norm, y_norm)))
    
    # 添加所有节点
    for index, (x, y) in gaze_data_normalized:
        positions[index] = (x, y)
        G.add_node(index, pos=(x, y))
    
    # 添加边，包括方向和距离
    for i in range(len(gaze_data_normalized)):
        index1, (x1, y1) = gaze_data_normalized[i]
        for j in range(max(0, i-k), min(len(gaze_data_normalized), i+k+1)):
            if i != j:
                index2, (x2, y2) = gaze_data_normalized[j]
                
                # 计算归一化后的距离和方向
                distance = calculate_distance(x1, y1, x2, y2)
                orientation = calculate_orientation(x1, y1, x2, y2)
                
                G.add_edge(index1, index2, weight=distance, orientation=orientation)
                
    with open(output_name, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G

# 构建单眼的k-hop STGG图，包含方向信息
def draw_graph2(gaze_data, k, output_name, max_nodes=100):
    G = nx.DiGraph()

    # 对 x 和 y 进行归一化
    step = len(gaze_data) / float(max_nodes)
    indices = [int(step * i) for i in range(max_nodes)]
    sampled_gaze_data = [gaze_data[i] for i in indices]
    gaze_data_normalized = [(i, frame, (x / 1920.0, y / 1080.0)) for i, (frame, x, y) in enumerate(sampled_gaze_data)]
    
    # 添加所有节点
    for index, frame, (x, y) in gaze_data_normalized:
        G.add_node(index, pos=(x, y), frame_number=frame)
    
    # 添加边，包括方向和距离
    for i in range(len(gaze_data_normalized)):
        index1, frame1, (x1, y1) = gaze_data_normalized[i]
        for j in range(max(0, i-k), min(len(gaze_data_normalized), i+k+1)):
            if i != j:
                index2, frame2, (x2, y2) = gaze_data_normalized[j]
                # 计算归一化后的距离和方向
                distance = calculate_distance(x1, y1, x2, y2)
                orientation = calculate_orientation(x1, y1, x2, y2)
                G.add_edge(index1, index2, weight=distance, orientation=orientation)
                
    with open(output_name, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G

import torch
def draw_graph3(gaze_data, k, output_name, max_nodes=50):
    node_features = []
    time_features = []
    edge_features = []
    edges = []
    
    # 对 x 和 y 进行归一化
    step = len(gaze_data) / float(max_nodes)
    indices = [int(step * i) for i in range(max_nodes)]
    sampled_gaze_data = [gaze_data[i] for i in indices]
    
    # 处理节点特征和时间特征
    for i, (frame, x, y) in enumerate(sampled_gaze_data):
        # normalized_x = x / 1920.0
        # normalized_y = y / 1080.0
        normalized_x = x / 240.0
        normalized_y = y / 240.0
        node_features.append([normalized_x, normalized_y])
        time_features.append(frame)
    
    # 处理边特征
    for i in range(len(sampled_gaze_data)):
        index1, frame1, x1, y1 = i, sampled_gaze_data[i][0], sampled_gaze_data[i][1], sampled_gaze_data[i][2]
        for j in range(max(0, i-k), min(len(sampled_gaze_data), i+k+1)):
            if i != j:
                index2, frame2, x2, y2 = j, sampled_gaze_data[j][0], sampled_gaze_data[j][1], sampled_gaze_data[j][2]
                distance = calculate_distance(x1, y1, x2, y2)
                orientation = calculate_orientation(x1, y1, x2, y2)
                edges.append([index1, index2])
                edge_features.append([distance, orientation])
    
    # 将节点特征、边特征和边索引保存为Tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)
    time_features = torch.tensor(time_features, dtype=torch.float32)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 将数据保存到文件
    # with open(output_name, 'wb') as f:
    #     pickle.dump((node_features, time_features, edge_features, edge_index), f, protocol=pickle.HIGHEST_PROTOCOL)
    return node_features, time_features, edge_features, edge_index

     
def draw_graph4(gaze_data, k, output_name, max_nodes=50):
    node_features = []
    time_features = []
    edge_features = []
    edges = []
    
    # 对数据进行采样
    step = len(gaze_data) / float(max_nodes)
    indices = [int(step * i) for i in range(max_nodes)]
    sampled_gaze_data = [gaze_data[i] for i in indices]
    
    # 处理节点特征和时间特征，同时记录有效节点
    valid_nodes = []  # 存储有效节点的索引
    node_validity = []  # 标记节点是否有效
    
    for i, (frame, x, y) in enumerate(sampled_gaze_data):
        # 归一化坐标
        normalized_x = x / 240.0
        normalized_y = y / 240.0
        
        # 检查是否在有效范围内
        is_valid = (0 <= normalized_x <= 8) and (0 <= normalized_y <= 4.5)
        node_validity.append(is_valid)
        
        # 只添加有效节点
        if is_valid:
            node_features.append([normalized_x, normalized_y])
            time_features.append(frame)
            valid_nodes.append(i)  # 记录有效节点的原始索引
    
    # 创建原始索引到新索引的映射
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_nodes)}
    
    # 处理边特征 - 只连接有效节点
    for i in range(len(sampled_gaze_data)):
        # 如果当前节点无效，跳过
        if not node_validity[i]:
            continue
            
        index1, frame1, x1, y1 = i, sampled_gaze_data[i][0], sampled_gaze_data[i][1], sampled_gaze_data[i][2]
        
        # 只考虑有效范围内的相邻节点
        for j in range(max(0, i-k), min(len(sampled_gaze_data), i+k+1)):
            # 跳过自身或无效节点
            if i == j or not node_validity[j]:
                continue
                
            index2, frame2, x2, y2 = j, sampled_gaze_data[j][0], sampled_gaze_data[j][1], sampled_gaze_data[j][2]
            
            # 计算距离和方向
            distance = calculate_distance(x1, y1, x2, y2)
            orientation = calculate_orientation(x1, y1, x2, y2)
            
            # 使用映射后的新索引
            new_i = index_mapping[i]
            new_j = index_mapping[j]
            edges.append([new_i, new_j])
            edge_features.append([distance, orientation])
    
    # 将节点特征、边特征和边索引保存为Tensor
    node_features = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.empty((0, 2), dtype=torch.float32)
    time_features = torch.tensor(time_features, dtype=torch.float32) if time_features else torch.empty((0), dtype=torch.float32)
    edge_features = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 2), dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    
    return node_features, time_features, edge_features, edge_index
   
# 绘制图，包括方向信息
def plot_graph(G, title):
    pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(4, 4))  # 正方形尺寸更容易观察归一化后的分布
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1)  # 边的颜色和粗细
    nx.draw_networkx_nodes(G, pos, node_color='teal', edgecolors='black', node_size=30)  # 节点颜色和大小
    
    plt.xlim(0, 1)  # X轴范围设定为[0, 1]
    plt.ylim(1, 0)  # 翻转 Y 轴的范围
    plt.xlabel('X coordinate', fontsize=8)
    plt.ylabel('Y coordinate', fontsize=8)
    plt.title(title, fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)  # 调整刻度的字体大小
    plt.gca().set_aspect('equal', adjustable='box')  # 保持 X 和 Y 轴比例一致
    plt.show()
    
# 保存图
def save_graph(G, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# 主程序
# if __name__ == "__main__":
#     file_path = 'path_to_your_csv.csv'  # CSV文件路径
#     k = 5  # k-hop 参数
#     data = read_data(file_path)
    
#     # 为左眼生成图
#     G_left = build_single_eye_k_stgg(data, k, 'left')
#     plot_graph(G_left, 'Left Eye k-STGG with Orientation')
#     # save_graph(G_left, 'left_eye_k_stgg_with_orientation_graph.gpickle')
    
#     # 为右眼生成图
#     G_right = build_single_eye_k_stgg(data, k, 'right')
#     plot_graph(G_right, 'Right Eye k-STGG with Orientation')
#     # save_graph(G_right, 'right_eye_k_stgg_with_orientation_graph.gpickle')