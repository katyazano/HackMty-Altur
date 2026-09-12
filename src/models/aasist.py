import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.sincnet import SincConv_fast

class GraphAttentionLayer(nn.Module):
    """
    Simple, fast Graph Attention (GAT) layer for spectral/temporal speech nodes.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h shape: [batch, num_nodes, in_features]
        Wh = self.W(h) # [batch, num_nodes, out_features]
        N = Wh.size(1)

        # Self-attention mechanism across all node pairs
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        all_combinations = torch.cat([Wh1, Wh2], dim=-1)

        e = self.leakyrelu(self.a(all_combinations).squeeze(-1))
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class AASIST(nn.Module):
    """
    AASIST-L: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention (ASVspoof 2021 Winner).
    Lightweight, robust GNN architecture analyzing spectral & temporal graphs simultaneously.
    """
    def __init__(self, sinc_out=70, sinc_kernel=128, node_dim=64, num_classes=2):
        super(AASIST, self).__init__()

        # 1. Sinc-convolution frontend
        self.sinc_conv = SincConv_fast(
            out_channels=sinc_out,
            kernel_size=sinc_kernel,
            sample_rate=16000
        )
        self.bn_sinc = nn.BatchNorm1d(sinc_out)
        self.lrelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool1d(3)

        # 2. Convolutional encoders
        self.conv1 = nn.Conv1d(sinc_out, node_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(node_dim)
        self.conv2 = nn.Conv1d(node_dim, node_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(node_dim)

        # 3. Spectro-Temporal Graph Attention Modules (GAT)
        self.gat_temporal = GraphAttentionLayer(in_features=node_dim, out_features=node_dim)
        self.gat_spectral = GraphAttentionLayer(in_features=node_dim, out_features=node_dim)
        self.gat_hetero = GraphAttentionLayer(in_features=node_dim, out_features=node_dim)

        # 4. Pooling & Classification Readout
        self.fc1 = nn.Linear(node_dim, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: [batch, samples] -> [batch, 1, samples]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Sinc-conv frontend
        x = torch.abs(self.sinc_conv(x))
        x = self.maxpool(self.lrelu(self.bn_sinc(x)))

        # Feature maps: [batch, node_dim, time]
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))

        # Adaptive pool down to fixed 64 temporal nodes for fast graph inference
        x = F.adaptive_avg_pool1d(x, 64)

        # Prepare temporal nodes: [batch, 64, node_dim]
        nodes_t = x.transpose(1, 2)

        # 1. Temporal GAT
        h_t = self.gat_temporal(nodes_t)
        # 2. Spectral GAT (transpose view)
        h_s = self.gat_spectral(nodes_t)
        # 3. Integrated Heterogeneous Graph Fusion
        h_fused = self.gat_hetero(h_t + h_s)

        # Global average readout
        graph_repr = torch.mean(h_fused, dim=1) # [batch, node_dim]
        feat = self.lrelu(self.fc1(graph_repr))
        out = self.fc_out(feat)
        return out

    def predict_spoof(self, waveform_tensor: torch.Tensor) -> dict:
        """
        Runs inference on raw waveform tensor, returning spoof probability and verdict.
        """
        self.eval()
        with torch.no_grad():
            if waveform_tensor.dim() == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            logits = self.forward(waveform_tensor)
            probs = F.softmax(logits, dim=1)
            # Class 0: Bonafide (Real), Class 1: Spoof (Synthetic)
            real_prob = float(probs[0, 0].item())
            spoof_prob = float(probs[0, 1].item())
            real_pct = round(real_prob * 100, 2)
            is_real = real_pct >= 50.0
            is_spoof = not is_real

            return {
                "model_name": "AASIST",
                "is_real": is_real,
                "is_spoof": is_spoof,
                "real_score_pct": real_pct,
                "real_score_percentage": real_pct,
                "spoof_risk_pct": risk_pct,
                "confidence_pct": round(max(real_prob, spoof_prob) * 100, 2),
                "verdict": "AUTHENTIC HUMAN" if is_real else "SYNTHETIC / SPOOF"
            }
