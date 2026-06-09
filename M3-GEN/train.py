import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import MultiModalDiseasePredictor 

# ================= 配置 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = '/root/autodl-tmp/death/processed_Heart_Failure.pt' 
SAVE_DIR = 'checkpoints_hf'
BATCH_SIZE = 64         
EPOCHS = 50              # 轮数
LEARNING_RATE = 2e-5     # 学习率
PATIENCE = 8             # 早停耐心值

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 1. 定义 Dataset (修复导入错误) =================
class MIMICDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        # 加载处理好的 .pt 文件
        data = torch.load(data_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32) # [N, 4, 24]
        self.y = torch.tensor(data['y'], dtype=torch.float32) # [N]
        self.texts = data['texts']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 1. 生理信号处理
        # 假设模型需要多视角输入，4 个通道切分成 [2, 2]
        
        x_full = self.X[idx]      # [4, 24]
        x_view1 = x_full[0:2, :]  # [2, 24]
        x_view2 = x_full[2:4, :]  # [2, 24]
        x_list = [x_view1, x_view2]

        # 2. 文本处理
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_inputs = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 3. 模拟人口统计学数据 (因为之前的预处理没保存这个，这里生成两个虚拟特征防止报错)
        
        demo_inputs = torch.tensor([0.5, 0.5], dtype=torch.float32) 

        label = self.y[idx].unsqueeze(0) # [1]
        
        return x_list, text_inputs, demo_inputs, label

# ================= 2. 定义 Mask 生成器 =================
class MaskGenerator:
    def __init__(self):
        pass
    
    def generate_mask(self, x, mode='random', ratio=0.2):
        """
        x: [Batch, Channels, Time]
        """
        mask = torch.ones_like(x)
        B, C, T = x.shape
        
        if mode == 'random':
            # 随机 Mask 掉 ratio 比例的点
            prob = torch.rand_like(x)
            mask[prob < ratio] = 0
            
        elif mode == 'block':
            # 块状缺失 (模拟传感器一段时间掉线)
            block_len = int(T * ratio)
            if block_len > 0:
                start = np.random.randint(0, T - block_len)
                mask[:, :, start:start+block_len] = 0
                
        return mask

# ================= 3. 早停工具 (Early Stopping) =================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# ================= 4. 损失函数 (带自动权重) =================
def compute_loss(y_pred, y_true, E_signal, Z_semantic, mu, logvar, pos_weight):
    # 1. 任务 Loss (带权重，解决样本不平衡)
    y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
    
    loss_bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    
    # 动态加权：正样本 loss * pos_weight
    weight_vector = y_true * pos_weight + (1 - y_true) * 1.0
    task_loss = (loss_bce * weight_vector).mean()

    # 2. 对比损失 & KL 散度 (正则化)
    # 对比损失实现
    if E_signal is not None and Z_semantic is not None:
        z1 = F.normalize(E_signal, dim=1)
        z2 = F.normalize(Z_semantic, dim=1)
        contrastive_loss = 1 - F.cosine_similarity(z1, z2).mean()
    else:
        contrastive_loss = torch.tensor(0.0).to(y_pred.device)
    
    if mu is not None and logvar is not None:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= y_pred.size(0)
    else:
        kl_loss = torch.tensor(0.0).to(y_pred.device)
    
    return task_loss + 0.1 * contrastive_loss + 0.01 * kl_loss

# ================= 5. 验证函数 =================
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in loader:
            x_list, text_inputs, demo_inputs, y_true = batch
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            demo_inputs = demo_inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)
            
            # 验证时不加 Mask
            x_processed_list = []
            for x in x_list:
                x = x.to(DEVICE)
                # 拼接全 1 的 Mask 表示无缺失
                mask = torch.ones_like(x).to(DEVICE)
                x_concat = torch.cat([x, mask], dim=1)
                x_processed_list.append(x_concat)

            y_hat, _, _, _, _ = model(x_processed_list, text_inputs, demo_inputs)
            loss = criterion(y_hat, y_true)
            total_loss += loss.item()
            
            all_probs.extend(y_hat.cpu().numpy())
            all_labels.extend(y_true.cpu().numpy())
            
    try:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except:
        auc, auprc = 0.5, 0.0
        
    return total_loss / len(loader), auc, auprc


def add_attention_visualization(model, val_loader, device, save_dir):
    """
    训练完成后添加注意力可视化分析
    """
    from visualization import AttentionVisualizer, analyze_attention_patterns
    
    print("\n🎨 Generating Attention Visualizations...")
    
    # 创建可视化器
    visualizer = AttentionVisualizer(save_dir=os.path.join(save_dir, 'attention_plots'))
    
    # 分析注意力模式
    results = analyze_attention_patterns(
        model, val_loader, device, visualizer, num_samples=20
    )
    
    attentions = results['attentions']
    predictions = results['predictions']
    labels = results['labels']
    
    # 1. 绘制单个样本的热力图
    for i in range(min(5, len(attentions))):
        visualizer.plot_attention_heatmap(
            attentions[i], 
            sample_id=i,
            title=f'Sample {i} - Pred: {predictions[i][0]:.3f}, True: {int(labels[i][0])}'
        )
    
    # 2. 按预测结果分组分析
    # 高风险组 vs 低风险组
    high_risk_idx = predictions.flatten() > 0.7
    low_risk_idx = predictions.flatten() < 0.3
    
    if np.sum(high_risk_idx) > 0 and np.sum(low_risk_idx) > 0:
        high_risk_attn = np.mean(attentions[high_risk_idx], axis=0)
        low_risk_attn = np.mean(attentions[low_risk_idx], axis=0)
        
        visualizer.plot_attention_comparison(
            [low_risk_attn, high_risk_attn],
            ['Low Risk', 'High Risk'],
            'Attention Patterns by Risk Level'
        )
    
    # 3. 按真实标签分组分析
    deceased_idx = labels.flatten() == 1
    survived_idx = labels.flatten() == 0
    
    if np.sum(deceased_idx) > 0 and np.sum(survived_idx) > 0:
        deceased_attn = np.mean(attentions[deceased_idx], axis=0)
        survived_attn = np.mean(attentions[survived_idx], axis=0)
        
        visualizer.plot_attention_comparison(
            [survived_attn, deceased_attn],
            ['Survived', 'Deceased'],
            'Attention Patterns by Outcome'
        )
    
    # 4. 绘制注意力流向图（选择几个典型样本）
    for i in range(min(3, len(attentions))):
        visualizer.plot_attention_flow(
            attentions[i], 
            sample_id=i,
            threshold=0.3
        )
    
    # 5. 统计摘要
    print("\n📊 Attention Pattern Summary:")
    print(f"Average attention matrix across all samples:")
    avg_attn = np.mean(attentions, axis=0)
    print(avg_attn)
    
    # 计算模态主导性
    modality_names = visualizer.modality_names
    for i, name in enumerate(modality_names):
        self_attn = avg_attn[i, i]
        cross_attn_in = np.mean([avg_attn[j, i] for j in range(len(modality_names)) if j != i])
        cross_attn_out = np.mean([avg_attn[i, j] for j in range(len(modality_names)) if j != i])
        
        print(f"\n{name} Modality:")
        print(f"  - Self-attention: {self_attn:.3f}")
        print(f"  - Received from others: {cross_attn_in:.3f}")
        print(f"  - Sent to others: {cross_attn_out:.3f}")
    
    print(f"\n✅ Attention visualizations saved to {visualizer.save_dir}")
    
    return visualizer, results

# ================= 主程序 =================
if __name__ == "__main__":
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到文件 {DATA_PATH}")
        print("请先运行数据预处理代码，生成 .pt 文件！")
        exit()

    print(f"🚀 Loading Data: {DATA_PATH}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 初始化 Dataset
    full_dataset = MIMICDataset(DATA_PATH, tokenizer)
    
    # 1. 划分数据集 (8:1:1)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    print(f"📚 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # 保存测试集供 test.py 使用
    torch.save(test_data, os.path.join(SAVE_DIR, 'test_dataset.pt'))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # # 2. 自动计算类别权重 (Pos Weight)
    # y_samples = []
    # sample_scan_limit = min(len(train_data), 2000)
    # print(f"正在扫描前 {sample_scan_limit} 个样本计算正负比例...")
    
    count_pos = 0
    for i in range(sample_scan_limit):
        # Dataset[i] 返回的是 tuple，最后一个是 label
        label = train_data[i][-1] 
        count_pos += int(label.item())
        
    count_neg = sample_scan_limit - count_pos
    
    # 防止分母为 0
    pos_weight_val = count_neg / (count_pos + 1e-5)
    POS_WEIGHT = torch.tensor(pos_weight_val).to(DEVICE)
    
    print(f"⚖️  Class Balance (Sample): Pos={count_pos}, Neg={count_neg}")
    print(f"⚖️  Auto Pos Weight: {pos_weight_val:.4f}")

    # 3. 初始化模型
    # 注意：这里 in_channels_list 对应 Dataset 里的切分，这里是 [2, 2]
    # 如果你 Dataset 里没切分，这里要改成 [4]
    model = MultiModalDiseasePredictor(
        in_channels_list=[2, 2], 
        hidden_dim=64,
        num_classes=1,
        demo_dim=2
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    mask_gen = MaskGenerator()

    # 4. 训练循环
    print("\n🔥 Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x_list, text_inputs, demo_inputs, y_true = batch
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            demo_inputs = demo_inputs.to(DEVICE)
            y_true = y_true.to(DEVICE)
            
            # Mask Augmentation Logic
            x_processed_list = []
            rand_val = np.random.random()
            if rand_val < 0.3: mode, ratio = 'block', np.random.uniform(0.1, 0.5)
            elif rand_val < 0.5: mode, ratio = 'random', np.random.uniform(0.1, 0.5)
            else: mode, ratio = 'random', 0.0
            
            for x in x_list:
                x = x.to(DEVICE)
                mask = mask_gen.generate_mask(x, mode=mode, ratio=ratio).to(DEVICE)
                x_masked = x * mask
                x_concat = torch.cat([x_masked, mask], dim=1)
                x_processed_list.append(x_concat)
            
            optimizer.zero_grad()
            y_hat, E_sig, Z_sem, mu, logvar = model(x_processed_list, text_inputs, demo_inputs)
            
            loss = compute_loss(y_hat, y_true, E_sig, Z_sem, mu, logvar, POS_WEIGHT)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        val_loss, val_auc, val_auprc = evaluate(model, val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")
        
        early_stopping(val_loss, model, os.path.join(SAVE_DIR, 'model_final.pth'))
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    print("\n✅ Training Finished.")
        # 添加注意力可视化
    visualizer, attention_results = add_attention_visualization(
        model, val_loader, DEVICE, SAVE_DIR
    )
