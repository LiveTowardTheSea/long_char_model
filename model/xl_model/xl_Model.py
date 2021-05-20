from xl_Encoder import *
from CRF_Decoder import *
import torch
from utils import *
import torch.nn as nn
import psutil
import os


class xl_Model(nn.Module):
    def __init__(self, config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size):
        super(xl_Model, self).__init__()
        self.config = config
        self.encoder = xl_Encoder(config, src_embedding_num, embedding_matrix, embedding_dim_size)
        self.decoder = CRF_decoder(config.d_model*2, tag_num)
    
    def get_encoder_output(self,src,mask,use_gpu):
        # faltten
        reverse_mask = (mask == False)
        src_ = torch.masked_select(src, reverse_mask)
        src_ = src_.unsqueeze(0)
        # 拿到 输出
        pad_idx = 1
        encoder_output = self.encoder(src_, pad_idx, use_gpu)
        # 将encoder_output 还原出来 batch 的维度，然后放到 decoder里生成输出
        # (batch_size, seq_len,d_model*2)
        feature_vec = torch.randn(reverse_mask.size()[0], reverse_mask.size()[1], self.config.d_model*2,device='cuda')
        sentence_len_tensor = torch.sum(reverse_mask, dim=-1).long()
        begin_offset = 0
        for i in range(sentence_len_tensor.shape[0]):
            feature_vec[i][0:sentence_len_tensor[i]] = encoder_output[begin_offset: begin_offset+ sentence_len_tensor[i]]
            begin_offset += sentence_len_tensor[i]
        del encoder_output
        return feature_vec


    def forward(self, src, y, mask, use_gpu):
        """

        :param src:(当前分段句子个数，seq_len)
        :param y: (当前分段句子个数，seq_len)
        :param mask:(当前分段句子个数，seq_len)
        :param use_gpu:是否使用gpu
        :return:
        """
        #print('模型计算之前：{:.8f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
        feature_vec = self.get_encoder_output(src, mask, use_gpu)
        #print('计算encoder之后当前进程的内存使用：{:.8f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
        loss, path = self.decoder.loss(feature_vec, y, mask, use_gpu)
        #print('计算decoder之后进程的内存使用：{:.8f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
        return loss, path