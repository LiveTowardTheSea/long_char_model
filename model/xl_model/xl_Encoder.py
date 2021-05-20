import torch
import torch.nn as nn
import copy
from xl_EncoderLayer import *
from util import *
import psutil
import os

# 双向的 共用一个 embedding 层，但是不共用上下文编码器
# 传过来真正的序列，我们将其倒叙，分别传给对应的单向 上下文编码器
class xl_Encoder(nn.Module):
    def __init__(self, config, src_embedding_num, embedding_matrix, embedding_dim_size):
        super(xl_Encoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(src_embedding_num, embedding_dim_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 按照Attention is all you need的方法，生成一个位置向量
        # 下面，按照 sinusoid encoding matrix 生成一个相对位置编码的向量
        self.relative_encoding = sinusoid_pos_encode((self.config.reuse_seg_num + 1) * self.config.seg_len,
                                                     config.k_dim)
        self.relative_encoding.requires_grad = False
        # 我们有两个encoder，正向和逆向，
        self.encoder = one_xl_Encoder(config, embedding_module=self.embedding,name='right')
        self.reverse_encoder = one_xl_Encoder(config, embedding_module=self.embedding,name='reverse')

    def forward(self, src, pad_idx, use_gpu):
        """

        :param src: (1,总长度)
        :param pad_idx:
        :param use_gpu:是否使用 gpu,
        :return:(context_len,2*d_model)
        """
        # 为逆向的transformer 生成输入
        reverse_src = torch.flip(src, dims=[1])
        result = self.encoder(src, pad_idx, self.relative_encoding, use_gpu)
        reverse_result_ = self.reverse_encoder(reverse_src, pad_idx, self.relative_encoding, use_gpu)
        reverse_result = torch.flip(reverse_result_, dims=[0])
        final_result = torch.cat([result, reverse_result], dim=-1)
        return final_result


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 单向的，使用 xl_Encoder传过来的embedding
class one_xl_Encoder(nn.Module):
    def __init__(self, config, embedding_module,name):
        super(one_xl_Encoder, self).__init__()
        self.config = config
        self.embedding = embedding_module
        if name == 'right':
            self.layer_num = config.first_layer_num
        else:
            self.layer_num = config.second_layer_num
        self.encoder_layers = get_clones(xl_EncoderLayer(config), self.layer_num)
        # 对于每一个方向的 encoder,我们训练一个u,v
        self.u_vec = nn.Parameter(torch.randn(1, config.k_dim))
        self.v_vec = nn.Parameter(torch.randn(1, config.k_dim))

    def forward(self, src, pad_idx, R, use_gpu):
        """
        :param src: (1,context_len) 有可能跨越多个分段，或者不足一个分段。
        对于不满足的我们在最后一个分段进行pad
        :param pad_idx 我们补全分段的pad
        :param R 相对位置编码，我们要对其进行倒叙，然后传递至每一个自层
        :param use_gpu 模型是否使用gpu
        :return: (context_len,d_model)
        """
        # 我们 首先对 位置编码进行倒叙处理，
        #print('计算encoder之前当前进程的内存使用：{:.8f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
        relative_encoding = torch.flip(R, dims=[0])  # ((reuse_seg+1)*seg_len,k_dim)
        # 我们会利用 之前几个分段的一些状态进行状态重用，我们初始化一个 previous向量进行记录
        # 在layer_num这一个维度上，第i行 代表为了第i个隐藏层准备的状态
        if use_gpu:
            relative_encoding = relative_encoding.cuda()
        previous = torch.zeros(self.config.reuse_seg_num, self.layer_num,
                               self.config.seg_len, self.config.d_model)
        # 使用gpu的话
        if use_gpu:
            previous = previous.cuda()
        # 接下来，我们将输入进行补全
        # 我们需要一个记录当前段所有信息的tensor, 在当前段处理结束之后，我们来更改 previous 向量
        record = torch.empty((self.layer_num, self.config.seg_len, self.config.d_model))
        if use_gpu:
            record = record.cuda()

        seq_len = src.size()[-1]
        if seq_len % self.config.seg_len != 0:
            temp = torch.tensor([pad_idx] * (self.config.seg_len - seq_len % self.config.seg_len), dtype=torch.long)
            temp = temp.unsqueeze(0)
            if use_gpu:
                temp = temp.cuda()
            src = torch.cat((src, temp), dim=-1)
        src = src.view(1, -1, self.config.seg_len)  # (1, seg_num, seg_len)
        # 接下来，我们在分段上进行时序处理
        # 我们需要一个记录每一个分段数据在最后一层的结果
        result = torch.empty(src.size()[1], self.config.seg_len, self.config.d_model)
        if use_gpu:
            result = result.cuda()
        for i in range(src.size()[1]):
            init_seq = src[:, i, :]
            # state (1,seg_len,d_model)
            state = self.embedding(init_seq)
            # 当attend 的时候，前面可能并没有我们所想象的那么多分段，那么这个时候我们便把他mask掉
            # mask 形状是 (seg_len, (reuse_seg_num + 1) * seg_num)
            mask = generate_mask(self.config.seg_len, self.config.reuse_seg_num, i)
            if use_gpu:
                mask = mask.cuda()
            #  接着，我们在层数上进行处理，在最后一层时，就是当前分段的最后结果
            for j in range(self.layer_num):
                record[j] = state.squeeze(0)
                # last_src (1，max_len, d_model)
                last_src = previous[:, j, :, :].contiguous().view(-1, previous.size()[-1]).unsqueeze(0)
                # 产生新状态
                state = self.encoder_layers[j](last_src, state, relative_encoding, self.u_vec, self.v_vec, mask,
                                               use_gpu)
            # 在处理完当前分段之后，我们要进行previous左移，然后将当前分段各个层的结果放置在previous里
            previous[0: -1] = previous.clone()[1:]
            previous[-1] = record
            # 处理完当前的分段，我们要记录当前分段最后一层的结果，就可以返回了
            result[i] = state.squeeze(0)
        # 接下来，对不同的segment 进行view,生成一个新的向量
        result = result.view(-1, result.size()[-1])
        # 因为有pad的存在，所以我们截取回来原来的长度
        # del previous,record
        #print('encoder计算完之后后当前进程的内存使用：{:.8f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
        return result[0:seq_len]