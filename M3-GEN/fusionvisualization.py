import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from pathlib import Path
import networkx as nx

class AttentionVisualizer:
    def __init__(self, save_dir='attention_plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 4个模态的名称
        self.modality_names = ['HR', 'ABP', 'Resp', 'Text']
        self.num_modalities = 4
        
    def plot_attention_heatmap(self, attention_matrix, sample_id, title=None):
        """
        绘制单个样本的注意力热力图 (4x4)
        attention_matrix: [4, 4]
        """
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        sns.heatmap(attention_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   xticklabels=self.modality_names,
                   yticklabels=self.modality_names,
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(title or f'Cross-Modality Attention (Sample {sample_id})', fontsize=14)
        plt.xlabel('Target Modality', fontsize=12)
        plt.ylabel('Source Modality', fontsize=12)
        
        # 添加注释框
        plt.text(0.5, -0.15, 
                'Self-attention on diagonal\nCross-attention off-diagonal',
                transform=plt.gca().transAxes,
                fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # 保存图片
        save_path = self.save_dir / f'attention_heatmap_sample_{sample_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_attention_comparison(self, attention_matrices, labels, title):
        """
        比较不同组别的平均注意力模式
        attention_matrices: list of [4,4] matrices
        labels: group labels
        """
        n_groups = len(attention_matrices)
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for i, (attn_matrix, label) in enumerate(zip(attention_matrices, labels)):
            sns.heatmap(attn_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       xticklabels=self.modality_names,
                       yticklabels=self.modality_names,
                       vmin=0, vmax=1,
                       ax=axes[i],
                       cbar=i==n_groups-1,
                       cbar_kws={'label': 'Attention Weight'} if i==n_groups-1 else None)
            axes[i].set_title(f'{label}', fontsize=12)
            axes[i].set_xlabel('Target Modality')
            axes[i].set_ylabel('Source Modality')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        save_path = self.save_dir / f'attention_comparison_{title.replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_attention_flow(self, attention_matrix, sample_id, threshold=0.25):
        """
        绘制注意力流向图（有向图）
        """
        G = nx.DiGraph()
        
        # 添加节点
        for i, name in enumerate(self.modality_names):
            G.add_node(name, pos=None)
        
        # 添加边（权重高于阈值）
        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                if i != j and attention_matrix[i, j] > threshold:
                    G.add_edge(self.modality_names[i], 
                              self.modality_names[j], 
                              weight=attention_matrix[i, j])
        
        plt.figure(figsize=(12, 8))
        
        # 使用分层布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 绘制节点
        node_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        nx.draw_networkx_nodes(G, pos, node_color=node_colors[:len(G.nodes)], 
                              node_size=3000, alpha=0.8)
        
        # 绘制边（粗细表示权重）
        edges = G.edges()
        if edges:
            weights = [G[u][v]['weight'] * 8 for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                  width=weights, alpha=0.6,
                                  edge_color='red', 
                                  arrows=True, arrowsize=25,
                                  arrowstyle='->',
                                  connectionstyle='arc3,rad=0.1')
            
            # 添加边标签（权重值）
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' 
                          for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 添加自注意力标签
        for i, name in enumerate(self.modality_names):
            if name in pos:
                plt.text(pos[name][0], pos[name][1]-0.2, 
                        f'Self: {attention_matrix[i,i]:.3f}',
                        horizontalalignment='center', 
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        plt.title(f'Attention Flow Network (Sample {sample_id}, threshold={threshold})', 
                 fontsize=14, pad=20)
        plt.axis('off')
        
        save_path = self.save_dir / f'attention_flow_sample_{sample_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_modality_importance_bar(self, avg_attention):
        """
        绘制模态重要性条形图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 自注意力
        self_attn = [avg_attention[i, i] for i in range(self.num_modalities)]
        
        # 接收的注意力 (列平均，去掉自注意力)
        received = [np.mean([avg_attention[j, i] for j in range(self.num_modalities) if j != i]) 
                   for i in range(self.num_modalities)]
        
        # 发送的注意力 (行平均，去掉自注意力)
        sent = [np.mean([avg_attention[i, j] for j in range(self.num_modalities) if j != i]) 
               for i in range(self.num_modalities)]
        
        x = np.arange(self.num_modalities)
        width = 0.6
        
        # 自注意力条形图
        bars1 = axes[0].bar(x, self_attn, width, color='steelblue', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.modality_names)
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Self-Attention', fontsize=12)
        axes[0].set_ylim([0, 1])
        # 添加数值标签
        for bar, val in zip(bars1, self_attn):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.3f}', ha='center', fontsize=9)
        
        # 接收注意力条形图
        bars2 = axes[1].bar(x, received, width, color='coral', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.modality_names)
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Received from Others', fontsize=12)
        axes[1].set_ylim([0, 1])
        for bar, val in zip(bars2, received):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.3f}', ha='center', fontsize=9)
        
        # 发送注意力条形图
        bars3 = axes[2].bar(x, sent, width, color='lightgreen', alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(self.modality_names)
        axes[2].set_ylabel('Attention Weight')
        axes[2].set_title('Sent to Others', fontsize=12)
        axes[2].set_ylim([0, 1])
        for bar, val in zip(bars3, sent):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.3f}', ha='center', fontsize=9)
        
        plt.suptitle('Modality Importance Analysis', fontsize=14, y=1.05)
        plt.tight_layout()
        
        save_path = self.save_dir / 'modality_importance_bar.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

def analyze_attention_patterns(model, loader, device, visualizer, num_samples=50):
    """
    分析注意力模式与预测结果的关系
    """
    model.eval()
    all_attentions = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(all_attentions) >= num_samples:
                break
                
            x_list, text_inputs, demo_inputs, y_true = batch
            
            # 处理输入
            x_processed_list = []
            for x in x_list:
                x = x.to(device)
                mask = torch.ones_like(x).to(device)
                x_concat = torch.cat([x, mask], dim=1)
                x_processed_list.append(x_concat)
            
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            demo_inputs = demo_inputs.to(device)
            
            # 获取注意力权重
            y_hat, _, _, _, _, A_cmi = model(
                x_processed_list, text_inputs, demo_inputs, return_attention=True
            )
            
            # 收集数据
            all_attentions.append(A_cmi.cpu().numpy())
            all_predictions.append(y_hat.cpu().numpy())
            all_labels.append(y_true.numpy())
    
    # 转换为numpy数组
    all_attentions = np.concatenate(all_attentions, axis=0)[:num_samples]
    all_predictions = np.concatenate(all_predictions, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    return {
        'attentions': all_attentions,
        'predictions': all_predictions,
        'labels': all_labels
    }

def generate_attention_report(results, visualizer):
    """生成注意力分析报告"""
    
    attentions = results['attentions']
    predictions = results['predictions']
    labels = results['labels']
    
    # 创建报告文本
    report = []
    report.append("="*60)
    report.append("ATTENTION MECHANISM ANALYSIS REPORT")
    report.append("="*60)
    
    # 1. 整体统计
    report.append("\n1. OVERALL STATISTICS")
    report.append(f"   Total samples analyzed: {len(attentions)}")
    report.append(f"   Positive cases: {np.sum(labels)}")
    report.append(f"   Negative cases: {len(labels) - np.sum(labels)}")
    
    # 2. 平均注意力矩阵
    avg_attn = np.mean(attentions, axis=0)
    report.append("\n2. AVERAGE ATTENTION MATRIX")
    report.append("   " + " ".join([f"{m:6s}" for m in visualizer.modality_names]))
    for i, row in enumerate(avg_attn):
        report.append(f"   {visualizer.modality_names[i]}: " + 
                     " ".join([f"{x:.3f}" for x in row]))
    
    # 3. 模态重要性分析
    report.append("\n3. MODALITY IMPORTANCE ANALYSIS")
    
    for i, mod in enumerate(visualizer.modality_names):
        self_attn = avg_attn[i, i]
        received = np.mean([avg_attn[j, i] for j in range(visualizer.num_modalities) if j != i])
        sent = np.mean([avg_attn[i, j] for j in range(visualizer.num_modalities) if j != i])
        
        report.append(f"\n   {mod} Modality:")
        report.append(f"     - Self-attention: {self_attn:.3f}")
        report.append(f"     - Received from others: {received:.3f}")
        report.append(f"     - Sent to others: {sent:.3f}")
        
        # 判断角色
        if self_attn > 0.5:
            role = "主导模态 (Dominant)" if self_attn > received else "独立模态 (Independent)"
        elif sent > received:
            role = "信息源 (Information Source)"
        elif received > sent:
            role = "信息汇聚点 (Information Sink)"
        else:
            role = "平衡模态 (Balanced)"
        report.append(f"     - 角色: {role}")
    
    # 4. 预测置信度与注意力模式的相关性
    report.append("\n4. CORRELATION WITH PREDICTION CONFIDENCE")
    
    # 计算每个样本的注意力熵（表示注意力分散程度）
    attention_entropy = []
    for attn in attentions:
        flat_attn = attn.flatten()
        flat_attn = flat_attn / (np.sum(flat_attn) + 1e-8)
        entropy = -np.sum(flat_attn * np.log(flat_attn + 1e-8))
        attention_entropy.append(entropy)
    
    # 计算预测置信度（距离0.5的绝对距离）
    confidence = np.abs(predictions.flatten() - 0.5) * 2
    
    # 计算相关系数
    correlation = np.corrcoef(attention_entropy, confidence)[0, 1]
    report.append(f"   Correlation between attention entropy and confidence: {correlation:.3f}")
    
    # 5. 按结局分层的注意力模式
    report.append("\n5. ATTENTION PATTERNS BY OUTCOME")
    
    deceased_mask = labels.flatten() == 1
    survived_mask = labels.flatten() == 0
    
    if np.sum(deceased_mask) > 0 and np.sum(survived_mask) > 0:
        deceased_attn = np.mean(attentions[deceased_mask], axis=0)
        survived_attn = np.mean(attentions[survived_mask], axis=0)
        
        report.append("\n   Deceased Patients (Positive):")
        for i, row in enumerate(deceased_attn):
            report.append(f"     {visualizer.modality_names[i]}: " + 
                         " ".join([f"{x:.3f}" for x in row]))
        
        report.append("\n   Survived Patients (Negative):")
        for i, row in enumerate(survived_attn):
            report.append(f"     {visualizer.modality_names[i]}: " + 
                         " ".join([f"{x:.3f}" for x in row]))
        
        # 计算差异
        diff = deceased_attn - survived_attn
        report.append("\n   Difference (Deceased - Survived):")
        for i, row in enumerate(diff):
            report.append(f"     {visualizer.modality_names[i]}: " + 
                         " ".join([f"{x:+.3f}" for x in row]))
    
    # 保存报告
    report_path = visualizer.save_dir / 'attention_report.txt'
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    return report
