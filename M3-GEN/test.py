import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    average_precision_score,
    confusion_matrix
)
from model import MultiModalDiseasePredictor
from train import MIMICDataset, MaskGenerator 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = '/root/autodl-tmp/checkpoints_hf'

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    计算全套医疗指标
    y_true: 真实标签 [0, 1, 1, ...]
    y_prob: 预测概率 [0.1, 0.9, 0.8, ...]
    """
    # 将概率转换为 0/1 预测
    y_pred = [1 if p > threshold else 0 for p in y_prob]
    
    # 基础指标
    try:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob) # PR曲线面积，对不平衡数据极其重要
    except:
        auc, auprc = 0.5, 0.0

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": auc,
        "AUPRC": auprc,
        "Acc": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

def evaluate_with_ratio(model, dataloader, missing_ratio, mode='block'):
    model.eval()
    all_preds, all_labels = [], []
    mask_gen = MaskGenerator()
    
    with torch.no_grad():
        for batch in dataloader:
            x_list, text_inputs, demo_inputs, y_true = batch
            
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            demo_inputs = demo_inputs.to(DEVICE)
            
            x_processed_list = []
            for x in x_list:
                x = x.to(DEVICE)
                mask = mask_gen.generate_mask(x, mode=mode, ratio=missing_ratio).to(DEVICE)
                x_masked = x * mask
                x_concat = torch.cat([x_masked, mask], dim=1)
                x_processed_list.append(x_concat)
            
            y_hat, _, _, _, _ = model(x_processed_list, text_inputs, demo_inputs)
            all_preds.extend(y_hat.cpu().numpy())
            all_labels.extend(y_true.numpy())
            
    return calculate_metrics(all_labels, all_preds)

if __name__ == "__main__":
    # 1. 路径检查
    dataset_path = os.path.join(SAVE_DIR, 'test_dataset.pt')
    model_path = os.path.join(SAVE_DIR, 'model_final.pth')
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("❌ 缺少文件，请先运行 train.py")
        exit()

    # 2. 加载数据与模型
    test_data = torch.load(dataset_path)
    loader = DataLoader(test_data, batch_size=16)
    
    model = MultiModalDiseasePredictor(
        in_channels_list=[2, 2], hidden_dim=64, num_classes=1, demo_dim=2
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    
    # 3. 运行详细评估
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # 存储指标以便画图
    history = {"AUC": [], "AUPRC": [], "F1": [], "Recall": []}
    
    print("\n📉 全面压力测试报告 (Comprehensive Stress Test Report)")
    print("=" * 95)
    # 表头
    print(f"{'Miss Rate':<10} | {'AUC':<8} | {'AUPRC':<8} | {'Acc':<8} | {'F1-Score':<10} | {'Recall':<8} | {'Precision':<10}")
    print("-" * 95)
    
    for r in ratios:
        metrics = evaluate_with_ratio(model, loader, missing_ratio=r, mode='block')
        
        # 记录数据
        history["AUC"].append(metrics["AUC"])
        history["AUPRC"].append(metrics["AUPRC"])
        history["F1"].append(metrics["F1"])
        history["Recall"].append(metrics["Recall"])
        
        # 打印一行
        print(f"{r*100:>5.0f}%     | {metrics['AUC']:.4f}   | {metrics['AUPRC']:.4f}   | {metrics['Acc']:.4f}   | {metrics['F1']:.4f}     | {metrics['Recall']:.4f}   | {metrics['Precision']:.4f}")
    
    print("=" * 95)

    # 4. 绘图 (画两张子图：一张AUC/AUPRC，一张Recall/F1)
    plt.figure(figsize=(12, 5))
    
    # 子图1: 综合性能 (AUC & AUPRC)
    plt.subplot(1, 2, 1)
    plt.plot(ratios, history["AUC"], marker='o', label='ROC-AUC', linewidth=2)
    plt.plot(ratios, history["AUPRC"], marker='s', label='AUPRC (PR Area)', linewidth=2)
    plt.title('Overall Performance Decay')
    plt.xlabel('Signal Missing Rate')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 子图2: 临床实用性 (Recall & F1)
    plt.subplot(1, 2, 2)
    plt.plot(ratios, history["Recall"], marker='^', color='red', label='Recall (Sensitivity)', linewidth=2)
    plt.plot(ratios, history["F1"], marker='d', color='green', label='F1-Score', linewidth=2)
    plt.title('Clinical Utility Decay')
    plt.xlabel('Signal Missing Rate')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'comprehensive_metrics.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 详细图表已保存至: {save_path}")
    print("💡 提示: 关注 Recall 的变化，这是衡量能否捕捉到所有高危病人的关键。")
