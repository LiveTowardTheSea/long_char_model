import torch
import torch.nn as nn
import torch.nn.functional as F
import math
SELF_ATTEN_VALUE = -1e5


def batched_index_select(qk, sticker):
    """
    根 sticker 的排序 ，将 qk 组织起来
    :param qk: (batch,head,seq_len,k_dim)
    :param sticker: (batch,head,round,seq_len)
    :return: (batch,head,round,seq_len,k_dim)
    """
    last_dim = qk.shape[-1]
    round_num = sticker.shape[-2]
    sticker_ = sticker.view(sticker.size()[0], sticker.size()[1], -1)
    sticker_ = sticker_.unsqueeze(-1).expand(-1, -1, -1, last_dim)
    return torch.gather(qk, dim=-2, index=sticker_).view(qk.shape[0], qk.shape[1], round_num, -1, last_dim)


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def across_chunk_attend(chunk_vec, attend_chunk_num):
    # 返回 (batch,head,round,bucket,bucket_len*(1+2*attend_chunk_num),k_dim)
    left = [torch.cat((chunk_vec[:, :, :, i:, :, :], chunk_vec[:, :, :, 0:i, :, :]), dim=-3) for i in
            range(1, attend_chunk_num + 1)]
    right = [torch.cat((chunk_vec[:, :, :, i:, :, :], chunk_vec[:, :, :, 0:i, :, :]), dim=-3) for i in
             range(-1, -attend_chunk_num - 1, -1)]
    after_chunk = left + [chunk_vec] + right
    return torch.cat(after_chunk, dim=-2)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class LSHAttention(nn.Module):
    def __init__(self, round_num=4, head=4, k_dim=64, d_model=256, bucket_num=20, attend_chunk_num=1):
        super().__init__()
        self.round_num = round_num
        self.head = head
        self.k_dim = k_dim
        assert head * k_dim == d_model, 'k_dim * head != d_model'
        self.bucket_num = bucket_num
        # 根据输入得到
        self.multi_head_q = nn.Linear(d_model, d_model)
        self.multi_head_v = nn.Linear(d_model, d_model)
        # 用于 hash 的向量
        assert bucket_num % 2 == 0, '桶的数量不能被2整除'
        self.hash_vec = torch.randn(k_dim, bucket_num // 2, round_num)
        self.hash_vec.requires_grad = False
        self.attend_chunk_num = attend_chunk_num
        self.output_matrix = nn.Linear(d_model,d_model)

    def get_hash(self, qk):
        # (batch,head,seq_len,bucket_num//2,round_num)
        # 返回(batch,head,round,seq_len)
        rotated_vec = torch.matmul(qk, self.hash_vec)
        rotated_vec = torch.cat([-rotated_vec, rotated_vec], dim=-2)
        score, buckets = torch.max(rotated_vec, dim=-2)
        return buckets.permute(0, 1, 3, 2)

    def forward(self, x, input_mask):
        # input_mask (batch,seq_len) 为 pad 的时候为True
        #  x (batch_size, seq_len, d_model)
        batch_size = x.shape[0]
        qk = self.multi_head_q(x)
        v = self.multi_head_v(x)
        # (batch,head,seq_len,k_dim)
        seq_len = qk.size()[1]
        qk = qk.view(qk.size()[0], qk.size()[1], self.head, self.k_dim).permute(0, 2, 1, 3)
        v = v.view(v.size()[0], v.size()[1], self.head, self.k_dim)
        buckets = self.get_hash(qk)
        ticker = torch.arange(0, seq_len)
        buckets = buckets * seq_len + ticker
        # s_ 意味着已经排好序的
        # s_ticker 意味着：当前存放的在原来的idx 是多少
        s_buckets, s_ticker = torch.sort(buckets, dim=-1)
        undo_sort = s_ticker.sort(dim=-1)[-1]  # (batch,head,round,seq_len) 第 i个在排序好的第几个里面
        # 不知道为什么，现在我们需要detach出来这些向量
        s_buckets = s_buckets.detach()
        s_ticker = s_ticker.detach()
        undo_sort = undo_sort.detach()
        # 我们根据 s_ticker 的顺序，将qk  v 靠拢在一起
        # (batch,head,round,seq_len,k_dim)
        s_qk = batched_index_select(qk, s_ticker)
        s_v = batched_index_select(v, s_ticker)
        # 接下来，将 k进行
        # 根据 chunk 进行 attend
        s_q = s_qk
        s_k = F.normalize(s_qk, p=2, dim=-1)
        assert seq_len // self.bucket_num == 0, 'seq_len 必须可以整除 桶的数量'
        # 每一个 chunk的长度是多少
        chunk_size = seq_len // self.bucket_num
        # self.attend_chunk_num 根据这个，我们判断要attend 之前之后的几个块
        chunk_s_ticker = torch.reshape((s_ticker.size()[0],s_ticker.size()[1],s_ticker.size()[2], -1, chunk_size))
        chunk_s_q = torch.reshape((s_q.size()[0], s_q.size()[1], s_q.size()[2], -1, chunk_size, s_q.size()[-1]))
        chunk_s_k = torch.reshape((s_k.size()[0], s_k.size()[1], s_k.size()[2], -1, chunk_size, s_k.size()[-1]))
        chunk_s_v = torch.reshape((s_v.size()[0], s_v.size()[1], s_v.size()[2], -1, chunk_size, s_v.size()[-1]))
        # (batch,head,round,bucket,seq_len/bucket,k_dim)
        across_chunk_s_ticker = across_chunk_attend(chunk_s_ticker.unsqueeze(-1),self.attend_chunk_num).squeeze(-1)
        across_chunk_s_k = across_chunk_attend(chunk_s_k, self.attend_chunk_num)
        across_chunk_s_v = across_chunk_attend(chunk_s_v, self.attend_chunk_num)
        dots = torch.matmul(chunk_s_q, across_chunk_s_k.transpose(-1, -2)) * (self.k_dim ** -0.5)
        mask_max_neg = max_neg_value(dots)
        # 下面进行 pad,将变长序列中，pad位置给mask掉
        # dots (batch,head,round,bucket,len,across_len)
        if input_mask is not None:
            s_mask = torch.gather(input_mask.expand_as(-1, self.head, self.round_num, -1), dim=-1, index=s_ticker)
            chunk_s_mask = s_mask.view(s_mask.shape[0], s_mask.shape[1], s_mask.shape[2], -1, chunk_size)
            across_chunk_s_mask = across_chunk_attend(chunk_s_mask.unsqueeze(-1)).squeeze(-1)
            dots = dots.masked_fill(across_chunk_s_mask.unsqueeze(-2),value=mask_max_neg)
            del across_chunk_s_mask

        # mask 掉当前位置
        self_mask = (chunk_s_ticker.unsqueeze(-1) == across_chunk_s_ticker.unsqueeze(-2))
        dots = dots.masked_fill(self_mask, value=SELF_ATTEN_VALUE)
        del self_mask

        # mask掉不是一个桶的
        s_buckets = s_buckets // seq_len
        chunk_s_buckets = across_chunk_s_buckets = torch.reshape(s_buckets, (batch_size,self.head,self.round_num,-1,chunk_size))
        across_chunk_s_buckets = across_chunk_attend(chunk_s_buckets.unsqueeze(-1), self.attend_chunk_num).squeeze(-1)
        buckets_mask = (chunk_s_buckets.unsqueeze(-1) != across_chunk_s_buckets.unsqueeze(-2))
        dots = dots.masked_fill(buckets_mask, value=mask_max_neg)
        del buckets_mask
        # 对于重复mask
        # The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition.
        # 下面这种情况是 只要 处在对应的 chunk就算是，一次
        # 但这种情况 在计算次数的时候，有时候虽然在可以 attend的chunk中，但并不在一个桶中
        # loc = undo_sort // chunk_size  # (batch,head,round,seq_len) 代表当前在第几个 chunk里面
        # # 左边的要在第几个chunk里才可以attend
        # left_loc = [(loc + i) % chunk_size for i in range(1, self.attend_chunk_num+1)]
        # # 右边的要在第几个chunk里才可以attend左
        # right_loc = [(loc - i + chunk_size) % chunk_size for i in range(1, self.attend_chunk_num+1)]
        # locs = left_loc + [loc] + [right_loc]
        # locs = torch.cat(locs, dim=2).permute(0, 1, 3, 2)  # (batch,head,seq_len,(1+2*self.attend_num)*round)
        # # (batch, head, round, seq_len, (1+2*self.attend_num)*round)
        # s_locs = batched_index_select(locs, sticker=s_ticker)
        # chunk_s_locs = torch.reshape(s_locs, (batch_size, self.head, self.round_num, -1, chunk_size, s_locs.size()[-1]))
        # # (batch,head,round,bucket,across_len,(1+2*self.attend_num)*round)
        # across_chunk_s_locs = across_chunk_attend(chunk_s_locs,self.attend_chunk_num)
        # start_pos = 2*self.attend_chunk_num*self.round_num
        # chunk_s_locs_1 = chunk_s_locs[:, :, :, :, :, None, start_pos:start_pos+self.round_num]
        # chunk_s_locs_1 = chunk_s_locs_1.expand(chunk_s_locs_1.shape[:5]+(1+2*self.attend_chunk_num, self.round_num))
        # chunk_s_locs = chunk_s_locs_1.view(chunk_s_locs.shape)
        # # 下面开始计算
        # #(batch,head,round,bucket,len,across_len,(1+2*self.attend)*round_num
        # dup_counts = (chunk_s_locs[:, :, :, :, :, None, :] == across_chunk_s_locs[:, :, :, :, None, :, :])
        # 下面这种情况 我们按照桶来 进行 计算、
        loc = buckets // seq_len
        loc = loc.permute(0, 1, 3, 2)  # (batch,head,seq_len,round)
        s_loc = batched_index_select(loc, s_ticker)  # (batch,head,round,seq_len,round)
        chunk_s_loc = torch.reshape((batch_size, self.head, self.round_num, -1, chunk_size, s_loc.size()[-1]))
        across_chunk_s_loc = across_chunk_attend(chunk_s_loc, self.attend_chunk_num)
        dup_counts = (chunk_s_loc[:, :, :, :, :, None, :] == across_chunk_s_loc[:, :, :, :, None, :, :])
        dup_counts = chunked_sum(dup_counts, chunks=(self.head * batch_size))
        dup_counts = dup_counts.detach()
        assert dup_counts.shape == dots.shape
        dots = dots - torch.log(dup_counts + 1e-9)
        del dup_counts
        # 获得 softmax的值
        # (batch,head,round,bucket,len,across_len)
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        # 下面我们和value相乘 (batch,head,round,bucket,len,k_dim)
        vo = torch.matmul(dots, across_chunk_s_v)
        # (batch, head, round,seq_len,k_dim)
        vo = vo.reshape((batch_size, self.head, self.round_num, -1, vo.size()[-1]))
        # (batch,head,round,seq_len)
        dots_logsumexp = dots_logsumexp.view(batch_size, self.head, self.round_num, -1)
        # 下面，我们还原原来的输入顺序
        # 得到了正常顺序的 v 以及正常的 v 以及 在不同 round上的 z
        normal_vo = torch.gather(vo, dim=-2, index=undo_sort.unsqueeze(-1).expand(vo.shape))
        normal_dots_round = torch.gather(dots_logsumexp, dim=-1, index=undo_sort)
        normal_dots_round_logsumexp = torch.logsumexp(normal_dots_round, dim=-2, keepdim=True)
        normal_dots_round = torch.exp(normal_dots_round - normal_dots_round_logsumexp).type_as(normal_dots_round)
        normal_vo = normal_vo * normal_dots_round.unsqueeze(-1)
        normal_vo = normal_vo.sum(dim=2)  # (batch,head,seq_len,k_dim)
        vo = normal_vo.permute(0, 2, 1, 3).contiguous().view(batch_size,seq_len,-1)
        output = self.output_matrix(vo)

        return output, buckets