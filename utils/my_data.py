# 存储一些数据集 vocab embedding 构建数据集的办法
from Vocab import *
from function import *
import pandas
import numpy as np
from iterator_dataset import *
import os


class Data:
    def __init__(self, opt):
        self.opt = opt   # 命令行的一些操作参数
        self.char_vocab = Vocab('char')
        self.tag_vocab = Vocab('tag', is_tag=True)
        self.char_embedding = None  # 这个是借鉴的呜呜呜子
        # 不论如何，将 data 定义在这里
        self.char_embedding_dim = 256
        # 这个主要是用于存放 训练 验证 测试数据的迭代器。
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None

    # train_data 是一个,我们统一使用BIO格式, 这里 MSRA 使用了 BIO, 我们不再去判断别的，
    # 随着数据集的增加，这个肯定是要补充的啦
    def build_tag_vocab(self, train_data):
        """
        生成 tag 的 vocab,tag_vocab没有 unk_token,但是我给搞了一个pad,
        :param train_data:训练数据集，类型为 DataFrame,sentence 为 char list
        :return:
        """
        print('building tag vocab')
        for tag_list in train_data['label']:
            for tag in tag_list:
                self.tag_vocab.add(tag)
        self.tag_vocab.tag_add_pad()

    def build_char_vocab(self, train_data, dev_data, test_data):
        """
        根据数据集加载出不同的 vocab 以及 tag,然后我们生成数据集的时候就可以从文件直接生成迭代数据集
        :return:
        """
        print('building char vocab')
        for sentence in train_data['sentence']:
            for token in sentence:
                self.char_vocab.add(token)
        for sentence in test_data['sentence']:
            for token in sentence:
                self.char_vocab.add(token)
        for sentence in dev_data['sentence']:
            for token in sentence:
                self.char_vocab.add(token)

    # 用于首次运行模型，将vocab, tag_vocab, pretrained_embedding保存
    # 通过如下代码我们假装自己准备好了 除了数据集之外的东西
    def build_vocab_pipeline(self):
        if self.opt.load_data is None:
            train_data = get_data(self.opt.train)
            dev_data = get_data(self.opt.dev)
            test_data = get_data(self.opt.test)
            self.build_char_vocab(train_data, dev_data, test_data)
            self.build_tag_vocab(train_data)
            # 通过之前的步骤，在首次，我们已经加载好了词语向量。
            # 接下来，我们来加载中文字词向量
            self.load_char_pretrained_embedding('data/news_char_256.vec')
            # 接下来我们要保存这些东西。
            self.char_vocab.save(self.opt.save_data + os.sep + 'char_vocab')
            self.tag_vocab.save(self.opt.save_data + os.sep + 'tag_vocab')
            # 保存词向量。
            print('saving vector of char')
            pretrained_file_name = 'char_embedding_matrix_' + str(self.char_embedding_dim)
            np.save(self.opt.save_data + os.sep + pretrained_file_name, self.char_embedding)
        else:
            # 这种情况就是直接加载数据
            # 首先 加载两个词向量
            self.char_vocab.load(self.opt.load_data + os.sep + 'char_vocab')
            self.tag_vocab.load(self.opt.load_data + os.sep + 'tag_vocab')
            self.char_embedding = np.load(self.opt.load_data + os.sep + 'char_embedding_matrix_256.npy')
            self.char_embedding_dim = self.char_embedding.shape[1]

    def build_data(self):
        # 下面我们已经初始化了 词汇表 词向量
        self.build_vocab_pipeline()
        # 接下来，我们开始准备数据 然后初始化到当前的 data里面
        # 但是batch_size是存在模型的config里面的，记得传过来哦
        if self.opt.status.lower() == 'train':
            self.train_iter = data_iterator(self.opt.train, self.char_vocab, self.tag_vocab)
            self.dev_iter = data_iterator(self.opt.dev, self.char_vocab, self.tag_vocab)
        elif self.opt.status.lower() == 'test':
            self.test_iter = data_iterator(self.opt.test, self.char_vocab, self.tag_vocab)
        elif self.opt.status.lower() == 'decode':
            pass
        else:
            print('input error:train or test or decode')

    def load_char_pretrained_embedding(self, char_pretrained_path):
        self.char_embedding, self.char_embedding_dim = load_pretrained_embedding(char_pretrained_path, self.char_vocab)





