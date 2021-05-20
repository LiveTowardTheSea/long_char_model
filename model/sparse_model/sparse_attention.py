import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_index_select(qk, index):
    #  qk（batch,global_head,seq_len,k_dim)
    # index (global_head,key_len) # 对所有 batch 都是一样的
    batch_size = qk.size()[0]
    k_dim = qk.size()[-1]
    index_ = index.unsqueeze(0).unsqueeze(-1).expand((batch_size, -1, -1, index))
    return torch.gather(qk, dim=2, index=index_)




# 对于不同的 head 使用 不同的方案
class Sparse_Attention(nn.Module):
    def __init__(self, d_model, head, stride, local_head_num, k_dim, c):
        super().__init__()
        self.multi_head_query = nn.Linear(d_model,d_model)
        self.multi_head_key = nn.Linear(d_model,d_model)
        self.multi_head_value = nn.Linear(d_model,d_model)
        self.head = head
        self.stride = stride
        self.local_head_num = local_head_num
        self.k_dim = k_dim
        self.c = c
        self.output_matrix = nn.Linear(d_model,d_model)

    def forward(self, x, mask):
        # x(batch,seq_len,d_model)
        # mask (batch_size,seq_len)
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        assert seq_len % self.stride == 0, 'the sequence length must be divided by stride'
        query = self.multi_head_query(x).view(batch_size, seq_len, self.head_num, -1)
        key = self.multi_head_key(x).view(batch_size, seq_len, self.head_num, -1)
        value = self.multi_head_value(x).view(batch_size, seq_len, self.head_num, -1)
        # 对于部分头，只 attend local 信息，我们取出这部分头，
        # (batch,seq_len,local_head,k_dim)
        local_query = query[:, :, 0:self.local_head_num, :]
        local_key = key[:, :, 0:self.local_head_num, :]
        local_value = value[:, :, 0:self.local_head_num, :]
        # 进行分块
        # (batch,num,stride,local_head,k_dim)
        chunk_local_query = local_query.reshape((batch_size, -1, self.stride, self.local_head_num, local_query.shape[-1]))
        chunk_local_key = local_query.reshape((batch_size, -1, self.stride, self.local_head_num, local_key.shape[-1]))
        chunk_local_value = local_value.reshape((batch_size, -1, self.stride, self.local_head_num, local_value.shape[-1]))
        # 进行形状变换
        chunk_local_query = chunk_local_query.permute(0, 3, 1, 2, 4).contiguous()
        chunk_local_key = chunk_local_key.permute(0, 3, 1, 4, 2).contiguous()
        chunk_local_value = chunk_local_value.permute(0, 3, 1, 2, 4).contiguous()
        chunk_local_qk = torch.matmul(chunk_local_query, chunk_local_key) * (self.k_dim ** -0.5)  #  (batch, head, num, stride, stride)
        # 下面进行mask，对pad的位置进行mask。
        chunk_mask = mask.reshape((batch_size, -1, self.stride))
        chunk_local_qk = chunk_local_qk.masked_fill(chunk_mask.unsqueeze(1).unsqueeze(-2), -1e10)
        chunk_local_qk = F.softmax(chunk_local_qk, dim=-1)
        chunk_local_qkv = torch.matmul(chunk_local_qk, chunk_local_value)  # (batch, head, num, stride,k_dim)
        local_qkv = chunk_local_value.contiguous().view(batch_size, self.local_head_num, -1, self.k_dim)
        # 下面进行全局的attention 处理
        # j mod l: l-c l-c+1.... l-1
        # 当使用多个头的时候，使他们分别 attend长度为l之内的，长度为c的不同的位置
        # 这也就意味着 对于不同的头而言,上述式子的结果不同，但结果集合长度都为c
        # 因为不需要causal mask，所以对于一个头，每一个位置attend 的位置都相同，我们拿到这些位置
        # (global_head, c) 为每一个头要 attend 的位置索引，在第一个stride 的块中
        start = self.stride - (self.head-self.local_head_num) * self.c
        assert start >= 0, '都超出第一个位置了,可以但没必要'
        each_head_attend = torch.arange(start, self.stride).long().reshape(self.head-self.local_head_num, self.c)
        # (global,seq_len//stride,c)
        each_head_attend = each_head_attend.unsqueeze(1).expand(-1, seq_len//self.stride, -1)
        distance = torch.arange(0, seq_len, self.stride).long()
        each_head_attend = each_head_attend + distance[None, :, None]
        # ( global, seq_len//stride*c)
        each_head_attend = torch.reshape(each_head_attend, (self.head-self.local_head_num, -1))
        global_query = query[:, :, self.local_head_num:, :]
        global_key = key[:, :, self.local_head_num:, :]
        global_value = value[:, :, self.local_head_num:, :]
        # 根据每个head 所 attend 的位置索引，我们取出 key value
        global_query = global_query.permute(0, 2, 1, 3).contiguous()
        global_key = global_key.permute(0, 2, 1, 3).contiguous()  # (batch,global_head,seq_len,k_dim)
        global_value = global_value.permute(0, 2, 1, 3).contiguous()
        # (batch, global_head, key_len, k_Dim)
        global_attend_key = batched_index_select(global_key, each_head_attend)
        global_attend_value = batched_index_select(global_value, each_head_attend)
        global_attend_key = global_attend_key.permute(0, 1, 3, 2).contiguous()
        # 对于一个头的所有序列，attend的位置是一样的 需要得到的 mask (batch,global_head,key_len)
        global_mask_idx = each_head_attend.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size,-1)
        # (batch, global_head * key_len)
        global_mask = torch.gather(mask, dim=-1, index=global_mask_idx).\
            reshape((batch_size, self.head-self.local_head_num, -1))
        global_qk = torch.matmul(global_query,global_attend_key) * (self.k_dim ** -0.5)
        global_qk = global_qk.masked_fill(global_mask.unsqueeze(-2), value=-1e10)
        global_qk = F.softmax(global_qk, dim=-1)
        global_qkv = torch.matmul(global_qk, global_attend_value)  # (batch,global_head,seq_len,k_dim)
        # 将头链接起来
        qkv = torch.cat((local_qkv, global_qkv), dim=1)
        qkv = qkv.permute(0,2,1,3).contiguous().view(batch_size,seq_len,-1)
        return qkv