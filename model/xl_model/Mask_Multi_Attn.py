import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask_Multi_Attn(nn.Module):
    def __init__(self, head,d_model,k_dim,p_drop):
        super(Mask_Multi_Attn, self).__init__()
        self.head = head
        self.multi_query = nn.Linear(d_model, d_model)
        self.multi_key_embed = nn.Linear(d_model, d_model)
        self.multi_key_pos = nn.Linear(k_dim, k_dim)
        self.multi_value = nn.Linear(d_model, d_model)
        self.output_matrix = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)
        assert (k_dim == (d_model // head))
    def forward(self,last_src,src, R, u_vec, v_vec, mask, use_gpu):
        """
        上一个分段上一层的数据在参与当前段的计算时，只参与前向计算，不进行后向传播
        :param last_src:上一个分段 上一层的结果 (1，Max_len,d_model)
        :param src: 这一个分段上一层的结果 (1,seg_len,d_model)
        :param R:是一个固定的位置向量 （M+L,k_dim) M可以跨越多个分段
                  要注意的是，此时此刻传过来的应该是一个倒序的，也就是说 [0]代表相对位置 M+L-1
        :param u :对于所有位置都一样，query 的 pos encode (1,k_dim)
        :param v:对于所有位置都一样，query 的 pos_encode 用于query  key 的pos_encode
        :param mask:因为只有一个batch,这个mask是用于 不attend之后的位置的 (seg_len,max_len+seg_len)
        :param use_gpu:是否使用gpu
        :return: 当前分段 当前层的输出 （1，seg_len,d_model)
        """
        # 首先，我们对上一个分段 上一层的结果进行冻结，使其只进行前向传播，不参与梯度更新
        sg_last_sc = last_src.detach()
        # last_state 与上一个分段在长度上进行连结 （1,max_len+seg_len,d_model)
        last_state = torch.cat((sg_last_sc, src), dim=1)
        q = self.multi_query(src)  # (1，seg_len,d_model)
        k = self.multi_key_embed(last_state)
        v = self.multi_value(last_state)
        # (1,head,seg_len,k_dim)
        multi_q = q.view(q.size()[0], q.size()[1], self.head, -1).permute(0, 2, 1, 3)
        # (1,head,k_dim,max_len+seg_len)
        multi_k = k.view(k.size()[0], k.size()[1], self.head, -1).permute(0, 2, 3, 1)
        # (1,head,max_len+seg_len,k_dim)
        multi_v = v.view(v.size()[0], v.size()[1], self.head, -1).permute(0, 2, 1, 3)
        ac_query = multi_q + u_vec
        bd_query = multi_q + v_vec
        ac_term = torch.matmul(ac_query, multi_k)
        relative_encode = self.multi_key_pos(R)
        relative_encode = relative_encode.transpose(0, 1).contiguous()  # (k_dim, M+L)
        bd_term_ = bd_query.matmul(relative_encode)  # (1,head,seg_len,M+L)
        # 对于最后两个维度，在倒数第二个维度上，对于每一个head,向左移动 L-1, 一直到0
        bd_term = torch.zeros(bd_term_.size(),device='cuda')
        # 最大上下文长度
        seg_len = bd_term_.size()[2]
        max_context_len = relative_encode.size()[-1] - seg_len
        # 分段长度
        for idx in range(bd_term_.size()[2]):
            bd_term[:, :, idx, 0:max_context_len + 1 + idx] = bd_term_[:, :, idx, seg_len-1-idx:]
        atten_score = ac_term + bd_term  # (1,head,seg_len,max_len+seg_len)
        if mask is not None:
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1)
        atten_score = atten_score.masked_fill(mask, -1e10)
        atten_score = F.softmax(atten_score, dim=-1)
        qkv_result = atten_score.matmul(multi_v)  # (batch_size,head_num,q_seq_len,v_dim)
        qkv_result = qkv_result.permute(0, 2, 1, 3)
        qkv_result = qkv_result.contiguous().view(qkv_result.shape[0], qkv_result.shape[1], -1)
        qkv_result = self.output_matrix(qkv_result)
        qkv_result = self.dropout(qkv_result)
        return qkv_result