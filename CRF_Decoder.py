import torch.nn as nn
import torch
import math

# 按照batch版本来实现的。
# 按照论文 Neural Architectures for Named Entity Recognition 来实现的。
IMPOSSIBLE = -1e4


def log_sum_exp(x, dim=1):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(dim)[0]
    return max_score + (x - max_score.unsqueeze(dim)).exp().sum(dim).log()


def compute_mask(mask, use_gpu):
    """
    mask 为 true false，true代表是是pad,false代表不是，
    我们要返回同等形状的tensor,false的位置为1，true的位置为 0
    :param mask:
    :return:
    """
    # 这里居然是int 
    temp = torch.ones(mask.shape[0], mask.shape[1], dtype=torch.float32)
    if use_gpu:
        temp =temp.cuda()
    binary_mask = temp.masked_fill(mask,value=0.0)
    binary_mask = binary_mask
    return binary_mask

class CRF_decoder(nn.Module):
    def __init__(self, d_model, tag_num):
        super(CRF_decoder, self).__init__()
        self.tag_num = tag_num
        self.d_model = d_model
        self.feature2tag = nn.Linear(d_model, tag_num, bias=False)
        self.transition_matrix = nn.Parameter(torch.randn((tag_num + 2, tag_num + 2)))
        # transition_matrix[i][j]代表了 从i到j的转移
        self.SRT_IDX = tag_num
        self.END_IDX = tag_num + 1
        # 为什么这里要加data
        self.transition_matrix.data[:, self.SRT_IDX] = IMPOSSIBLE
        self.transition_matrix.data[self.END_IDX, :] = IMPOSSIBLE

    def compute_score(self, tag_vec, y, mask,use_gpu):
        """
       针对一个给定的序列，计算当前 S(X,y)
       :param tag_vec(batch_size,seq_len,tag_num)
       :param y:(batch_size,seq_len) 真实值
       :param mask:batch会有一些 pad_idx，不应该求得
       :return: 对于每个batch,返回当前序列所对应的score
       """
        batch_size = tag_vec.shape[0]
        time_step = tag_vec.shape[1]
        binary_mask = compute_mask(mask, use_gpu)
        # print('当前mask尺寸:',binary_mask.shape, binary_mask.dtype)
        result = torch.zeros(batch_size, dtype=torch.float32).unsqueeze(1)
        # 求一下初始的转移结果
        if use_gpu:
            result = result.cuda()
        start_trans = self.transition_matrix[self.SRT_IDX, :]
        #print('到开始的转移矩阵，尺寸：',start_trans.size())
        result += start_trans[y[:, 0]].unsqueeze(-1)
        for i in range(time_step):
            #  首先求emission score:
            emit = tag_vec[:, i, :]  # (batch_size, tag_num)
            # 这里可能会出现问题，因为当前可能会存在pad_idx，但我们在tag_vocab中并没有pad 这个置于最后面的idx
            # 所以我们在这里，我们首先，把 含有pad_idx的东西都变成是一个在其范围之内的东东
            current_time_batch = y[:, i].masked_fill(mask[:, i], value = 0).long()
            emit_ = torch.gather(emit, dim=1, index=torch.unsqueeze(current_time_batch, dim=1))
            result += emit_ * binary_mask[:, i].unsqueeze(1)
            # 接下来，求解 transition score
            trans = self.transition_matrix[y[:, i], :]  # y[:, i]的形状是[batch_size],中间有pad
            if i != time_step - 1:
                trans_ = torch.gather(trans, dim=1, index=torch.unsqueeze(y[:, i + 1], dim=1))
                result += trans_  * binary_mask[:, i + 1].unsqueeze(1)

        end_trans = self.transition_matrix[:, self.END_IDX]
        sentence_length = time_step - torch.sum(mask, dim=-1).long()
        end_token = torch.gather(y, dim=1, index=torch.unsqueeze(sentence_length - 1, dim=1)).squeeze(-1)
        result += end_trans[end_token].unsqueeze(-1)
        #print("当前sentence_score为：",result.squeeze(-1))
        return result

    def sum_of_sentence_score(self, tag_vec, mask, use_gpu):
        """
        计算分母：log sum_exp(S(X,y))
        :param tag_vec: (batch_size,seq_len,tag_num)
        :param mask: (batch_size,seq_len)
        :return: 对每个 batch 求结果，返回得分总和
        """
        batch_size = tag_vec.size()[0]
        seq_len = tag_vec.size()[1]
        initial_graph = torch.zeros(batch_size, self.tag_num)
        if use_gpu:
            initial_graph = initial_graph.cuda()
        initial_graph += self.transition_matrix[self.SRT_IDX, :-2]
        previous = initial_graph
        binary_mask = compute_mask(mask,use_gpu)
        tran_score = self.transition_matrix[:-2, :-2]   # (tag_num,tag_num)
        for i in range(seq_len):
            obj = tag_vec[:, i, :]  # (batch_size,tag_num)
            if i == seq_len - 1:
                mask_2 = torch.zeros(batch_size, 1, dtype=torch.float32)
            else:
                mask_2 = binary_mask[:, i + 1].unsqueeze(1)
            if use_gpu:
                mask_2 = mask_2.cuda()
            scores = torch.unsqueeze(previous + obj * binary_mask[:, i].unsqueeze(1), dim=-1) + \
                    tran_score.unsqueeze(0) * mask_2.unsqueeze(-1)  # (batch_size,tag_num,tag_num)
            previous = log_sum_exp(scores)
        end_score = self.transition_matrix[:-2, self.END_IDX]
        final_score = end_score.unsqueeze(0) + previous
        return log_sum_exp(final_score, dim=-1)

    def viterbi_decode(self, tag_vec, mask, use_gpu):
        """
        对每个batch计算最大路径
        :param tag_vec: (batch_size,seq_len,tag_num)
        :param mask: (batch_size,seq_len)
        :return: 返回 batch 个最大路径
        """
        batch_size = tag_vec.size()[0]
        seq_len = tag_vec.size()[1]
        tran_score = self.transition_matrix[:-2, :-2]  # (tag_num,tag_num)
        previous = torch.zeros(batch_size, self.tag_num, self.tag_num)
        if use_gpu:
            previous = previous.cuda()
        previous += self.transition_matrix[self.SRT_IDX, :-2]
        # 一个记录每一段分数的向量
        stage_score = torch.zeros(batch_size, seq_len, self.tag_num)
        # 一个记录当前步骤取最大值时上一个步骤idx的向量
        stage_idx = torch.full((batch_size, seq_len, self.tag_num), 0, dtype=torch.long)
        if use_gpu:
            stage_score = stage_score.cuda()
            stage_idx = stage_idx.cuda()
        # 已经累加了即将转移到下一个标签的转移分数，没有累加下一个标签的分数
        for i in range(seq_len):
            obj = tag_vec[:, i, :].unsqueeze(1)  # (batch_size,1,tag_num)
            # 需要判断下一步是不是 就是 mask 了
            scores = previous + obj  # (batch_size, tag_num, tag_num)
            score, idx = torch.max(scores, dim=1)
            stage_idx[:, i, :] = idx
            stage_score[:, i, :] = score
            previous = score.unsqueeze(-1) + tran_score

        # 下面，我们把焦点放在stage_score 和stage_idx上面
        sentence_length_idx = seq_len - torch.sum(mask, dim=1).long() - 1
        final_token_score = torch.zeros(batch_size, self.tag_num)
        if use_gpu:
            final_token_score = final_token_score.cuda()
        for i in range(batch_size):
            final_token_score[i] = stage_score[i, sentence_length_idx[i], :]
        final_token_score = final_token_score + self.transition_matrix[:-2, self.END_IDX].unsqueeze(0)
        max_idx = torch.max(final_token_score, dim=1)[1]  # 最后一个token是什么
        path = []
        
        for i in range(batch_size):
            best_path = []
            now = max_idx[i]
            best_path.append(now.cpu().item())
            for j in range(sentence_length_idx[i], 0, -1):
                last = stage_idx[i, j, now].cpu().item()
                best_path.insert(0, last)
                now = last
            path.append(best_path)
        return path

    def loss(self, feature_vec, y, mask, use_gpu):
        """返回[batch_size]"""
        tag_vec = self.feature2tag(feature_vec)
        # 这里太大了，我们除以一个东西
        tag_vec = tag_vec / math.sqrt(self.d_model)
        s_xy = self.compute_score(tag_vec, y, mask, use_gpu)
        sum_sentence = self.sum_of_sentence_score(tag_vec, mask, use_gpu)
        #print("当前batch，所有路径总结果为：",sum_sentence)
        with torch.no_grad():
            path_ = self.viterbi_decode(tag_vec, mask, use_gpu)
        return sum_sentence - s_xy.squeeze(-1),path_

    def forward(self, feature_vec, mask, use_gpu):
        tag_vec = self.feature2tag(feature_vec)
        tag_vec = tag_vec / math.sqrt(self.d_model)
        best_path = self.viterbi_decode(tag_vec, mask, use_gpu)
        return best_path


