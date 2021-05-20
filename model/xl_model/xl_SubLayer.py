import torch
import torch.nn as nn
import torch.nn.functional as F
class Position_Wise_Network(nn.Module):
    # FCNN encoder的第二层
    def __init__(self, d_model, d_ff, p_drop):
        super(Position_Wise_Network, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    # 在预测是也要使用的哦
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 突然想起来这么初始化会不会不太对
        self.gamma = nn.Parameter(torch.randn(d_model))
        self.beta = nn.Parameter(torch.randn(d_model))
        self.eps = eps

    def forward(self, x):
        # x (batch_size,seq_len,d_model)
        x = (x - torch.mean(x, dim=-1, keepdim=True))/(torch.std(x, dim=-1, keepdim=True) + self.eps)
        norm = x * self.gamma + self.beta
        return norm
