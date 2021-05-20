# 以后可能会有存储有 char bichar word 等一些的字符表
# 每一个都会在数据集上有一个类似字典的东西，
import json
import codecs
import os
import traceback

class Vocab:
    def __init__(self, vocab_type, is_tag=False):
        self.vocab_type = vocab_type
        self.vocab_list = []  # 所有的字列表
        self.itos = {}  # idx --> string
        self.stoi = {}  # string->idx
        self.unk_token = '</unk>'  # 未知单词
        self.pad_token = '</pad>'
        self.unk_idx = 0
        self.pad_idx = 1
        self.init_vocab_dict(is_tag)

    def init_vocab_dict(self, is_tag):
        if not is_tag:
            self.itos[self.unk_idx] = self.unk_token
            self.stoi[self.unk_token] = self.unk_idx
            self.itos[self.pad_idx] = self.pad_token
            self.stoi[self.pad_token] = self.pad_idx

    def show_vocab(self):
        print('the vocab:', self.vocab_type, 'vocab size：', len(self.vocab_list))

    def get_length(self):
        return len(self.itos)

    def add(self, new_token):
        # 添加到当前的词典
        if new_token not in self.vocab_list:
            self.vocab_list.append(new_token)
            current_idx = len(self.itos)
            self.itos[current_idx] = new_token
            self.stoi[new_token] = current_idx

    def tag_add_pad(self):
        current_idx = len(self.itos)
        self.pad_idx = current_idx
        self.itos[current_idx] = self.pad_token
        self.stoi[self.pad_token] = current_idx

    def load(self, file_path):
        """"
        想要加载的无外乎三个 vocab_list itos stoi
        """
        # 但是如果这样加载的话，不管是itos 还是stoi ,其中的编号都是字符级别的。
        try:
            with codecs.open(file_path + os.sep + "vocab_list.txt", 'r', 'utf-8') as f1:
                self.vocab_list = json.load(f1)
            with codecs.open(file_path + os.sep + "itos.txt", 'r', 'utf-8') as f2:
                self.itos = json.load(f2)
            with codecs.open(file_path + os.sep + "stoi.txt", 'r', 'utf-8') as f3:
                self.stoi = json.load(f3)
            # 这时的 itos 经过序列化反序列化，key已经变为str类型，我们需要将其变为int类型
            itos = {}
            for key,value in self.itos.items():
                itos[int(key)] = value
            self.itos = itos
            if 'tag' in self.vocab_type:
                self.pad_idx = len(self.itos) - 1
        except Exception:
            print('loading vocab error')
            traceback.print_exc()
            exit(0)

    def save(self, file_path):
        # 首次进入 可能没有该文件夹呢。
        print('saving vocab_file: ', self.vocab_type)
        try:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            with codecs.open(file_path + os.sep + "vocab_list.txt", 'w', 'utf-8') as f1:
                json.dump(self.vocab_list, f1, ensure_ascii=False)
            with codecs.open(file_path + os.sep + "itos.txt", 'w', 'utf-8') as f2:
                json.dump(self.itos, f2, ensure_ascii=False)
            with codecs.open(file_path + os.sep + "stoi.txt", 'w', 'utf-8') as f3:
                json.dump(self.stoi, f3, ensure_ascii=False)
        except Exception:
            print('save vocab error')
            traceback.print_exc()
            exit(0)
