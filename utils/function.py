# #应该具备的功能：读取原始数据为csv文件，之后就直接加载csv文件。
import pandas as pd
import codecs
import traceback
import os
import numpy as np


def get_data(input_path):
    """
    数据首先是存储在txt中，我们要把他放到目录下的csv文件中，利用pandas
    :param input_path: 文件路径
    :return: 返回一个DataFrame类型的数据，一列是字符列表，另一列是idx 列表
    """
    df_data = pd.DataFrame(columns=['sentence', 'label'])
    print('getting data: ',input_path)
    try:
        with codecs.open(input_path, 'r', 'utf-8') as f:
            string_list = []
            tag_list = []
            for line in f.readlines():
                if line not in ['\n', '\r\n'] and not(line.strip().startswith("#") and line.strip().endswith("#")):
                    word_label = line.strip().split()
                    if len(word_label) >= 2:
                        string_list.append(word_label[0])
                        tag_list.append(word_label[1])
                elif line in ['\n', '\r\n']:
                    df_data.loc[df_data.index.size] = [string_list, tag_list]
                    string_list = []
                    tag_list = []
        return df_data
    except Exception as e:
        print(input_path, "file reading error")
        print(len(df_data))
        traceback.print_exc()
        exit(0)


# path应该初始化为一个值
def load_pretrained_embedding(pretrained_path, vocab):
    print('constructing pretrained embedding matrix')
    # 这里，向量应该是对应的一行一个单词 一堆词向量
    pretrained_embedding_dict, pretrained_embedding_dim = load_pretrained_emb(pretrained_path)
    print(pretrained_embedding_dim)
    oov_num = 0
    # 还要有pad 以及unk 的 embedding
    pretrained_vector = np.zeros((len(vocab.vocab_list)+2, pretrained_embedding_dim))
    pretrained_vector[vocab.unk_idx] = np.random.uniform(0, 1, pretrained_embedding_dim)
    pretrained_idx = vocab.pad_idx + 1
    for token in vocab.vocab_list:
        if token in pretrained_embedding_dict.keys():
            pretrained_vector[pretrained_idx] = pretrained_embedding_dict[token]
        else:
            oov_num += 1
            pretrained_vector[pretrained_idx] = np.random.uniform(0, 1, pretrained_embedding_dim)
        pretrained_idx += 1
    print('the vocab:', vocab.vocab_type, 'oov num:', oov_num)
    return pretrained_vector, pretrained_embedding_dim


def load_pretrained_emb(pretrained_path=None):
    """
    主要是从训练好的txt文件中，加载出 单词->embedding 的映射，返回维度
    :param pretrained_path: txt 文件的路径
    :return: dict{单词-->embedding},维度
    """
    print("loading pretrained embedding dict")
    pretrained_embedding_dict = {}
    pretrained_embedding_dim = 0
    try:
        with codecs.open(pretrained_path, 'r', 'utf-8') as f:
            for line in f.readlines():
                if len(line) >= 2:
                    word_vec = line.strip().split()
                    word = word_vec[0]
                    vec = []
                    for str_dim in word_vec[1:]:
                        vec.append(float(str_dim))
                    pretrained_embedding_dict[word] = vec
            pretrained_embedding_dim = len(vec)
    except Exception:
        print('vector txt file reading error!')
        traceback.print_exc()
        exit(0)

    return pretrained_embedding_dict, pretrained_embedding_dim
