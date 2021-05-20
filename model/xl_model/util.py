import torch
import math


def sinusoid_pos_encode(pos_len, k_dim):
    pos_tensor = torch.zeros(pos_len, k_dim)
    for pos in range(pos_len):
        for i in range(k_dim):
            if i % 2 == 0:
                pos_tensor[pos][i] = math.sin(pos / (10000 ** (2 * i / k_dim)))
            else:
                pos_tensor[pos][i] = math.cos(pos / (10000 ** (2 * i / k_dim)))
    return pos_tensor


def generate_mask(seg_len, reuse_num, current_seg):
    """
    生成mask,不能pad的位子设为True
    因为有时候我们所在的分段是在文章前几个的位子，没有那么多前置段，我们要把他们mask掉
    :param seg_len: 分段长度
    :param reuse_num: 重复使用的分段个数
    :param current_seg: 当前是在该输入的地几个分段，从0开始
    :return: mask (seg_len, (reuse_num +1) * seg_len)
    """
    current_seg_mask = torch.ones((seg_len, seg_len), dtype=torch.bool)
    current_seg_mask = torch.triu(current_seg_mask,diagonal=1)
    reuse_seg_mask = torch.zeros((seg_len, reuse_num * seg_len), dtype=torch.bool)
    if current_seg - reuse_num < 0:
        reuse_seg_mask[:, 0:(reuse_num-current_seg) * seg_len] = True
    result = torch.cat((reuse_seg_mask, current_seg_mask), dim=-1)
    return result
