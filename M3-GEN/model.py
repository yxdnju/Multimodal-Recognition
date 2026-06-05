import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# ==========================================
# 1. 基础组件
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, t = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class TemporalConvNet(nn.Module):
    """两层时序卷积网络 + SE注意力"""
    def __init__(self, in_channels, hidden_dims=[64, 128]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.se1 = SEBlock(hidden_dims[0])
        
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.se2 = SEBlock(hidden_dims[1])
        
    def forward(self, x):
        # 第一层
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1_tilde = self.se1(h1)
        
        # 第二层
        h2 = F.relu(self.bn2(self.conv2(h1_tilde)))
        h2_tilde = self.se2(h2)
        
        return h2_tilde


class AdaptiveWindowGRUCell(nn.Module):
    """自适应时序窗口GRU单元"""
    def __init__(self, input_dim, hidden_dim, max_window_len=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_window_len = max_window_len
        
        # 窗口系数计算
        self.w_omega = nn.Linear(hidden_dim + input_dim + hidden_dim, 1)
        self.b_omega = nn.Parameter(torch.zeros(1))
        
        # 注意力机制构建历史摘要
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # GRU门控
        self.W_z = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim + input_dim, hidden_dim)
        
    def compute_window_coefficient(self, h_prev, x_t, context_summary):
        """计算时序窗口系数 omega_t"""
        combined = torch.cat([h_prev, x_t, context_summary], dim=-1)
        omega_t = torch.sigmoid(self.w_omega(combined) + self.b_omega)
        return omega_t
    
    def build_adaptive_window_context(self, h_history, omega_t):
        """根据窗口系数构建自适应历史上下文摘要"""
        K_t = max(1, int(self.max_window_len * omega_t.item() if omega_t.dim() == 0 else omega_t.mean().item()))
        K_t = min(K_t, len(h_history))
        
        if K_t <= 0 or len(h_history) == 0:
            return torch.zeros_like(h_history[-1]) if len(h_history) > 0 else torch.zeros(1, self.hidden_dim).to(omega_t.device)
        
        h_window = torch.stack(h_history[-K_t:], dim=0)  # [K_t, B, H]
        
        # 注意力机制
        attn_weights = F.softmax(torch.matmul(self.attn_proj(h_window), h_window[-1:].transpose(0, 1)).squeeze(-1), dim=0)
        context = (attn_weights.unsqueeze(-1) * h_window).sum(dim=0)
        return context
    
    def forward(self, x_t, h_prev, h_history):
        # x_t: [B, input_dim], h_prev: [B, hidden_dim]
        # 先用简单平均值作为初始上下文摘要（实际迭代中需要循环）
        if len(h_history) > 0:
            simple_context = torch.stack(h_history, dim=0).mean(dim=0)
        else:
            simple_context = torch.zeros_like(h_prev)
        
        omega_t = self.compute_window_coefficient(h_prev, x_t, simple_context)
        
        # 构建自适应窗口上下文（简化版，实际需要存储更多历史）
        combined = torch.cat([h_prev, x_t], dim=-1)
        z_t = torch.sigmoid(self.W_z(combined))
        r_t = torch.sigmoid(self.W_r(combined))
        
        h_tilde = torch.tanh(self.W_h(torch.cat([r_t * h_prev, x_t], dim=-1)))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t, omega_t


class AdaptiveWindowGRU(nn.Module):
    """自适应时序窗口GRU（完整序列版本）"""
    def __init__(self, input_dim, hidden_dim, max_window_len=10):
        super().__init__()
        self.cell = AdaptiveWindowGRUCell(input_dim, hidden_dim, max_window_len)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x: [B, T, input_dim]
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        h_history = []
        outputs = []
        omega_list = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, omega_t = self.cell(x_t, h_t, h_history)
            h_history.append(h_t)
            outputs.append(h_t)
            omega_list.append(omega_t)
        
        return torch.stack(outputs, dim=1), torch.stack(omega_list, dim=1)


class MultiViewSignalEncoder(nn.Module):
    """多视角信号编码器（独立编码+掩码感知+自适应时序建模）"""
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        # 时序卷积网络提取局部特征
        self.tcn = TemporalConvNet(in_channels, hidden_dims=[hidden_dim // 2, hidden_dim])
        
        # 自适应时序窗口GRU
        self.aw_gru = AdaptiveWindowGRU(hidden_dim, hidden_dim)
        
        # 自注意力融合
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_with_mask):
        """
        x_with_mask: [B, C, T] 其中 C = 原始通道 + mask通道
        """
        # 1. 时序卷积网络提取局部特征
        local_feat = self.tcn(x_with_mask)  # [B, hidden_dim, T]
        
        # 2. 转置为 [B, T, hidden_dim] 用于GRU
        local_feat = local_feat.transpose(1, 2)  # [B, T, hidden_dim]
        
        # 3. 自适应时序窗口GRU
        sequential_out, _ = self.aw_gru(local_feat)  # [B, T, hidden_dim]
        
        # 4. 自注意力融合得到固定维度特征
        attn_out, _ = self.self_attn(sequential_out, sequential_out, sequential_out)
        final_feat = attn_out.mean(dim=1)  # [B, hidden_dim]
        
        return self.out_proj(final_feat)


# ==========================================
# 2. 跨模态交互融合模块
# ==========================================

class CrossModalAlignment(nn.Module):
    """跨模态语义对齐模块（对比对齐）"""
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, text_feat, signal_feat):
        """
        text_feat: [B, hidden_dim]
        signal_feat: [B, hidden_dim]
        返回对比损失
        """
        text_proj = F.normalize(self.proj(text_feat), dim=-1)
        signal_proj = F.normalize(self.proj(signal_feat), dim=-1)
        
        logits = torch.matmul(text_proj, signal_proj.T) / self.temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        loss_contrast = F.cross_entropy(logits, labels)
        return loss_contrast


class CrossModalInteraction(nn.Module):
    """跨模态交互模块（交叉注意力）"""
    def __init__(self, hidden_dim, num_modalities):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_feat, signal_feats):
        """
        text_feat: [B, hidden_dim]
        signal_feats: list of [B, hidden_dim]
        """
        # 将信号特征堆叠
        signal_stack = torch.stack(signal_feats, dim=1)  # [B, num_signals, hidden_dim]
        
        # 文本作为query，信号作为key/value
        text_unsqueezed = text_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        enhanced_text, _ = self.cross_attn(query=text_unsqueezed, key=signal_stack, value=signal_stack)
        enhanced_text = enhanced_text.squeeze(1)
        
        # 信号作为query，文本作为key/value
        enhanced_signals, _ = self.cross_attn(query=signal_stack, key=text_unsqueezed, value=text_unsqueezed)
        
        return enhanced_text, enhanced_signals


# ==========================================
# 3. 变分推理模块
# ==========================================

class VariationalInferenceModule(nn.Module):
    """变分推理模块（VAE + 重参数化）"""
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, fused_feat):
        mu = self.fc_mu(fused_feat)
        logvar = self.fc_logvar(fused_feat)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def kl_loss(self, mu, logvar):
        """KL散度损失：强制隐变量分布接近标准正态分布"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


# ==========================================
# 4. 多任务分类器
# ==========================================

class MultiTaskClassifier(nn.Module):
    """多任务分类器（主分类器 + 辅助分类器）"""
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))


# ==========================================
# 5. 主模型：面向缺失场景的鲁棒跨模态对齐网络
# ==========================================

class RobustCrossModalAlignmentNetwork(nn.Module):
    """
    面向缺失场景的鲁棒跨模态对齐网络 (RCMAN)
    Robust Cross-Modal Alignment Network for Missing Scenarios
    """
    def __init__(self, in_channels_list, hidden_dim, num_classes, 
                 text_dim=768, demo_dim=2, latent_dim=64, max_window_len=10):
        super().__init__()
        
        # 信号通道数（每个通道翻倍，因为要拼接mask）
        self.in_channels_list = [c * 2 for c in in_channels_list]
        self.num_signal_modalities = len(in_channels_list)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 5.2.1 多视角信号编码器（独立编码+掩码感知+自适应时序建模）
        self.signal_encoders = nn.ModuleList([
            MultiViewSignalEncoder(c, hidden_dim) for c in self.in_channels_list
        ])
        
        # 文本编码器（BioClinicalBERT + 分层微调）
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 人口统计学特征嵌入
        self.demo_proj = nn.Linear(demo_dim, hidden_dim)
        
        # 5.2.2 跨模态交互融合模块
        self.cross_align = CrossModalAlignment(hidden_dim)
        self.cross_interact = CrossModalInteraction(hidden_dim, self.num_signal_modalities)
        
        # 融合层
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # 文本+信号拼接后投影
        
        # 5.2.3 变分推理模块
        self.vae = VariationalInferenceModule(hidden_dim, latent_dim)
        
        # 多任务分类器
        self.main_classifier = MultiTaskClassifier(latent_dim, num_classes)
        self.text_aux_classifier = MultiTaskClassifier(hidden_dim, num_classes)
        self.signal_aux_classifier = MultiTaskClassifier(hidden_dim, num_classes)
        
    def apply_mask(self, signal, mask_ratio=0.15):
        """
        应用数据增强掩码（模拟缺失场景）
        signal: [B, 1, T]
        返回: masked_signal, mask_indicator
        """
        batch_size, channels, seq_len = signal.shape
        mask = torch.ones_like(signal)
        
        if self.training and mask_ratio > 0:
            # 随机掩码
            num_mask = int(seq_len * mask_ratio)
            for b in range(batch_size):
                mask_pos = torch.randperm(seq_len)[:num_mask]
                mask[b, :, mask_pos] = 0
        
        masked_signal = signal * mask
        return masked_signal, mask
    
    def forward(self, X_list, text_inputs, demo_inputs, mask_ratio=0.15, return_attention=False):
        """
        X_list: 信号列表 [B, 1, T] 每个元素对应 HR, ABP, Resp
        text_inputs: BERT输入
        demo_inputs: [B, 2] 年龄、性别
        """
        # ==========================================
        # 5.2.1 多视角信号编码
        # ==========================================
        signal_feats = []
        signal_raw_list = []
        
        for i, x in enumerate(X_list):
            # 应用掩码并拼接mask
            x_masked, mask_indicator = self.apply_mask(x, mask_ratio if self.training else 0)
            x_with_mask = torch.cat([x_masked, mask_indicator], dim=1)  # [B, 2, T]
            signal_raw_list.append(x_masked)
            
            # 多视角编码
            feat = self.signal_encoders[i](x_with_mask)  # [B, hidden_dim]
            signal_feats.append(feat)
        
        # ==========================================
        # 文本编码（分层微调：仅微调最后2层）
        # ==========================================
        if self.training:
            # 训练时：正常前向传播（BERT参数通过优化器更新）
            text_out = self.bert(**text_inputs)
        else:
            # 推理时：冻结BERT
            with torch.no_grad():
                text_out = self.bert(**text_inputs)
        
        text_cls = text_out.last_hidden_state[:, 0, :]  # [B, 768]
        text_feat = self.text_proj(text_cls)  # [B, hidden_dim]
        
        # ==========================================
        # 人口统计学特征
        # ==========================================
        demo_feat = self.demo_proj(demo_inputs)  # [B, hidden_dim]
        
        # 将文本和人口特征融合
        text_enhanced = text_feat + demo_feat
        
        # ==========================================
        # 5.2.2 跨模态交互融合
        # ==========================================
        # 对比对齐损失
        # 信号特征取平均作为整体信号表示
        signal_combined = torch.stack(signal_feats, dim=1).mean(dim=1)  # [B, hidden_dim]
        loss_contrast = self.cross_align(text_enhanced, signal_combined)
        
        # 交叉注意力交互
        enhanced_text, enhanced_signals = self.cross_interact(text_enhanced, signal_feats)
        
        # 拼接融合
        fused_feat = torch.cat([enhanced_text, enhanced_signals.mean(dim=1)], dim=-1)  # [B, 2*hidden_dim]
        fused_feat = self.fusion_proj(fused_feat)  # [B, hidden_dim]
        
        # ==========================================
        # 5.2.3 变分推理模块
        # ==========================================
        z, mu, logvar = self.vae(fused_feat)  # z: [B, latent_dim]
        loss_kl = self.vae.kl_loss(mu, logvar)
        
        # ==========================================
        # 多任务分类
        # ==========================================
        # 主分类器（使用隐变量z）
        y_hat_main = self.main_classifier(z)
        
        # 辅助分类器（仅文本 + 人口特征）
        y_hat_text = self.text_aux_classifier(text_enhanced)
        
        # 辅助分类器（仅信号）
        y_hat_signal = self.signal_aux_classifier(signal_combined)
        
        # 最终预测取主分类器结果
        y_hat = y_hat_main
        
        # 计算置信度
        sigma = torch.exp(0.5 * logvar).mean(dim=-1, keepdim=True)  # 标准差
        sigma_max = sigma.max() if sigma.max() > 0 else torch.ones_like(sigma)
        confidence = 1 - sigma / (sigma_max + 1e-8)
        
        if return_attention:
            return y_hat, confidence, {
                'main': y_hat_main,
                'text_aux': y_hat_text,
                'signal_aux': y_hat_signal,
                'mu': mu,
                'logvar': logvar,
                'loss_contrast': loss_contrast,
                'loss_kl': loss_kl,
                'sigma': sigma
            }
        else:
            return y_hat, confidence, {
                'main': y_hat_main,
                'text_aux': y_hat_text,
                'signal_aux': y_hat_signal,
                'loss_contrast': loss_contrast,
                'loss_kl': loss_kl
            }


# ==========================================
# 6. 损失函数
# ==========================================

class RCMANLoss(nn.Module):
    """
    组合损失函数：L = L_cls + α * L_aux + β * L_KL + γ * L_contrast
    """
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.1, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_coef = gamma
        self.focal_gamma = focal_gamma
        
    def focal_loss(self, pred, target):
        """Focal Loss for imbalanced classification"""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce_loss).mean()
    
    def forward(self, outputs, targets):
        """
        outputs: dict from model forward
        targets: [B, num_classes]
        """
        # 主分类损失
        loss_cls = self.focal_loss(outputs['main'], targets)
        
        # 辅助分类损失
        loss_text_aux = F.binary_cross_entropy(outputs['text_aux'], targets)
        loss_signal_aux = F.binary_cross_entropy(outputs['signal_aux'], targets)
        loss_aux = loss_text_aux + loss_signal_aux
        
        # KL散度损失
        loss_kl = outputs.get('loss_kl', 0)
        
        # 对比损失
        loss_contrast = outputs.get('loss_contrast', 0)
        
        # 组合损失
        total_loss = loss_cls + self.alpha * loss_aux + self.beta * loss_kl + self.gamma_coef * loss_contrast
        
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_aux': loss_aux,
            'loss_kl': loss_kl,
            'loss_contrast': loss_contrast
        }
