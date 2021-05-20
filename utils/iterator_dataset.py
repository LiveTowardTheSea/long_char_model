from numpy.lib.npyio import NpzFile
import torch
import codecs
# 迭代器功能：根据dataset、vocab 每次生成 （batch_size,seq_len)的torch数据，形成对于数据集的便利
import numpy as np
"""
功能：
    1.遍历整个数据集，形成返回下一个的迭代器
    2.返回  (batch_size,seq_len)  的 vocab_idx 及 tag_idx
        2.1 序列长度不一样，需要补全
        2.2 需要将 词语和标签 变成编号
"""


def generate_char_idx(batch_data, vocab):
    """
    这个是针对于 char 来说的，因为是只提取了单个的token
    :param batch_data: series类型的数据,有两列，要么是 sentence，要么是 label
    :param vocab: sentence 或者是label 的 vocab
    :return: idx 列表
    """
    idx = []
    for sentence in batch_data:
        each_idx = []
        for token in sentence:
            if token in vocab.vocab_list:
                each_idx.append(vocab.stoi[token])
            else:
                each_idx.append(vocab.unk_idx)
        idx.append(each_idx)
    return idx


def generate_pad_idx(idx_list, vocab):
    """
    经过上一个函数处理之后，我们可以把 token变为一系列的 idx,形成一个包含列表的列表，现在我们要做的就是
    把这个列表的列表变成tensor,并且补齐。
    :param idx_list:
    :param vocab: 进行补齐的vocab,里面含有vocab.pad
    :return: 返回 torch.tensor 以及 mask,根据pad
    """
    max_len = max([len(each_idx_list) for each_idx_list in idx_list])
    result = torch.zeros((len(idx_list), max_len), dtype=torch.long)
    for i, sentence in enumerate(idx_list):
        result[i] = torch.tensor(sentence + [vocab.pad_idx]*(max_len - len(sentence)), dtype=torch.long)
    mask = (result == vocab.pad_idx)
    return result, mask

    
def read_data(data_path):
    data_list = []
    seg_list = []
    with codecs.open(data_path, 'r', 'utf-8') as f:
        sentence_list = []
        tag_list = []
        for line in f.readlines():
            if line not in ['\n', '\r\n'] and not(line.strip().startswith("#") and line.strip().endswith("#")):
                word_label = line.strip().split()
                if len(word_label) >= 2:
                    sentence_list.append(word_label[0])
                    tag_list.append(word_label[1])
            elif line in ['\n', '\r\n']:
                if len(sentence_list)>0 and len(tag_list)>0 and len(sentence_list)==len(tag_list):
                    seg_list.append((sentence_list, tag_list))
                sentence_list = []
                tag_list = []
            elif line.strip().startswith("#") and line.strip().endswith("#"):
                data_list.append(seg_list)
                seg_list = []
                sentence_list = []
                tag_list = []
    return data_list


class data_iterator:
    def __init__(self, data_path, char_vocab, tag_vocab):
        print("init iterator:", data_path)
        self.data_list = read_data(data_path)
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.offset = 0
        self.max_num = len(self.data_list)

    def reset_iter(self):
        self.offset = 0
        # 重新洗一下数据
        np.random.shuffle(self.data_list)

    def next(self):
        if self.offset == self.max_num:
            raise StopIteration
        sentence_tag_data = self.data_list[self.offset]
        self.offset += 1
        sentence_data = []
        tag_data = []
        for sentence, tag in sentence_tag_data:
            sentence_data.append(sentence)
            tag_data.append(tag)
        sentence_idx = generate_char_idx(sentence_data, self.char_vocab)
        tag_idx = generate_char_idx(tag_data, self.tag_vocab)
        sentence_tensor, sentence_mask = generate_pad_idx(sentence_idx, self.char_vocab)
        tag_tensor, tag_mask = generate_pad_idx(tag_idx, self.tag_vocab)
        return sentence_tensor, tag_tensor, sentence_mask