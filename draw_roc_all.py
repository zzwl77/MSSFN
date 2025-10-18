import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_from_csv_selected_models(csv_file, selected_models, display_names, given_roc_auc_values, colors,
                                        title='Receiver operating characteristic example', save_path=None):
    """
    从 CSV 文件中读取数据，根据给定的模型列名列表进行筛选，并绘制多模型 ROC 曲线。
    
    参数：
    - csv_file: CSV 文件路径，第一列为真实标签，其余列为各模型的预测概率。
    - selected_models: list，包含要选择的模型在 CSV 中的列名称。
    - display_names: list，所选模型对应的显示名称，用于图例标注（可包含 LaTeX 语法）。
    - given_roc_auc_values: list，每个模型对应的 AUC 值（显示时直接采用，不重新计算）。
    - colors: list，用于区分各模型 ROC 曲线的颜色。
    - title: 图像标题。
    - save_path: 图像保存路径（例如 save_path=save_root），如果提供，则将图像保存到该路径。
    
    返回：
    - display_names（图例名称列表）和对应的给定的 ROC-AUC 值列表。
    """
    # 读取 CSV 数据
    df = pd.read_csv(csv_file)
    # 假设第一列为真实标签
    y_true = df.iloc[:, 0].values  

    # 绘制参考线（Chance Line）
    plt.plot([0, 1], [0, 1], 'k--')
    
    # 遍历每个选定模型
    for i, model in enumerate(selected_models):
        if model in df.columns:
            # y_score = df[model].values.astype(float)
            y_score = df[model].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) and '[' in x and ']' in x else x)

            # 计算 ROC 曲线（仅用于绘制曲线，AUC 值显示采用外部提供的）
            fpr, tpr, _ = roc_curve(y_true, y_score)
            color = colors[i % len(colors)]
            # 使用给定的 AUC 值进行显示
            plt.plot(fpr, tpr, color=color, 
                     label=f'{display_names[i]} (AUC={given_roc_auc_values[i]:.2f})')
        else:
            print(f"警告：CSV中未找到列名 {model}")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至：{save_path}")
    plt.show()
    
    return display_names, given_roc_auc_values


# ====================
# 使用示例：
# ====================
if __name__ == '__main__':
    # CSV文件路径，假设文件名为 'model_output.csv'
    csv_file_path = 'experiments/adncgcls/adncgcls.csv'
    
    # 用户提供要筛选的模型在 CSV 中的列名列表
    selected_models = ['biformer', 'transnext', 'rmt', 'dlieg', 'lagmf', 'mcgrl', 'adg6']
    
    # 对应的图例显示名称列表
    display_names = ['Biformer', 'TransNeXt', 'RMT', 'DLIEG', 'LAGMF', 'MCGRL', 'MSSFN']
    
    # 用户提供的各模型的 ROC-AUC 值列表（直接显示，不做重新计算）
    given_roc_auc_values = [0.78, 0.76, 0.77, 0.76, 0.75, 0.81, 0.87]
    
    # 自定义的颜色列表
    colors = [
        'peru', 'brown', 'deepskyblue', 'aquamarine', 'darkorange',
        'r', 'g', 'b', 'tan'
    ]
    
    # 假设保存路径由变量 save_root 给出
    save_root = 'experiments/adncgcls/roc_all_compare.png'
    
    # 调用函数绘制ROC曲线，并获得模型名称和给定的 ROC-AUC 值列表
    model_labels, roc_auc_values = plot_roc_from_csv_selected_models(
        csv_file=csv_file_path, 
        selected_models=selected_models, 
        display_names=display_names, 
        given_roc_auc_values=given_roc_auc_values,
        colors=colors,
        title='Receiver operating characteristic example',
        save_path=save_root
    )
    
    # 输出模型名称及其对应的ROC-AUC值
    print("模型名称：", model_labels)
    print("ROC-AUC值：", roc_auc_values)