import torch
import torch.nn as nn

from Mask_Multi_Attn import *
from xl_SubLayer import *


class xl_EncoderLayer(nn.Module):
    def __init__(self, config):
        super(xl_EncoderLayer, self).__init__()
        self.attention_layer = Mask_Multi_Attn(config.head, config.d_model, config.k_dim,config.p_drop)
        self.layer_norm = LayerNorm(config.d_model)
        self.fcnn = Position_Wise_Network(config.d_model, config.d_ff, config.p_drop)

    def forward(self, last_src, src, R, u_vec, v_vec, mask, use_gpu):
        temp = self.attention_layer(last_src, src, R, u_vec, v_vec, mask, use_gpu)
        src_ = src + temp
        src_ = self.layer_norm(src_)
        states = self.fcnn(src_)
        return states

