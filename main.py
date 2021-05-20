import sys
sys.path.append('utils')
sys.path.append('model/xl_model')
from xl_Model import *
from config import *
import my_data
import argparse
import torch
import torch.nn as nn
import time
import numpy as np
from utils.metric import *
from utils.iterator_dataset import *
import os
import sys
import psutil

def get_model(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size, use_gpu, load_model=None):
    if config.model_name == 'xl':
        model = xl_Model(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size)
    # elif config.model_name == 'universal':
    #     model = Universal_Model(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size)
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    else:
        for name_, p in model.named_parameters():
            if p.dim() > 1 and str("embedding") not in name_:
                nn.init.xavier_uniform_(p)
            print(name_, p.shape)
    print('需要训练的模型参数总量:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    if use_gpu:
        model_ = model.cuda()  # 不是就地操作
    return model_


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def decode(model, my_data, use_gpu):
    # 根据输入进行解码的句子，我们输入一个句子，然后进行解码
    model.eval()
    sentence = input("请输入您要识别的句子")
    # 接下来，我们根据词典，将句子变为 vocab 中的词语编号 转变为了tensor,下面我们进行解码
    sentence_list = list(sentence)
    sentence_list_idx = generate_char_idx([sentence_list], my_data.char_vocab)
    sentence_tensor, sentence_mask = generate_pad_idx(sentence_list_idx,my_data.char_vocab)
    # 放在gpu上面
    if use_gpu:
        sentence_tensor = sentence_tensor.cuda()
        sentence_mask = sentence_mask.cuda()
    # 下面，进行模型计算，我们得到了path 为列表的列表
    encoder_output = model.encoder(sentence_tensor, sentence_mask, use_gpu)
    path = model.decoder(encoder_output, sentence_mask, use_gpu)
    # 然后，我们在控制台输出结果
    # 因为我们只有一个句子，所以我们取第一个
    path = path[0]
    for i, tag_idx in enumerate(path):
        print(sentence_list[i], my_data.char_vocab.itos[tag_idx], end='\t')


def eval_model(model, my_data, mode='dev', use_gpu=True):
    """
       在控制台输出 句子总量 以及处理的总时间等
       用来评估模型，得到一系列的 值，例如 precision recall f acc 值
    :param model: 模型
    :param my_data: 存放数据集的东西
    :param mode: 是dev 还是test,我们要评估的数据集
    :param use_gpu: 是否使用gpu做训练 yeap
    :return: acc precision recall f
    """
    # 根据当前 batch 来进行累加，
    # 当前累加需要累加的一些对象
    right_num = 0  # 边界和种类都对的数量累加
    predict_num = 0  # 预测出的总共的实体数量
    golden_num = 0  # 真实值中实体的数量
    char_num = 0  # 总共的字符数
    acc_char_num = 0  # 标签匹配的字符数

    data_iter = None
    model.eval()
    if mode == 'dev':
        data_iter = my_data.dev_iter
    elif mode == 'test':
        data_iter = my_data.test_iter
    else:
        print('evaluation mode choose from test and dev')
    with torch.no_grad():
        while True:
            try:
                sentence_tensor, tag_tensor, sentence_mask = data_iter.next()
                # 转换到gpu上面
                if use_gpu:
                    sentence_tensor = sentence_tensor.cuda()
                    tag_tensor = tag_tensor.cuda()
                    sentence_mask = sentence_mask.cuda()
                encoder_output = model.get_encoder_output(sentence_tensor, sentence_mask, use_gpu)
                # decode_result 是一个 列表的列表，里面存放着当前预测的词汇编号
                decode_result = model.decoder(encoder_output, sentence_mask, use_gpu)
                # 是一个累加的问题 decode_result, tag_tensor为真实值
                # 我们先不写对于单个实体种类的计算问题，只涉及全部的种类。
                # 我们根据当前的 batch 进行累加
                cur_acc_num, cur_char_num, cur_right_num, cur_golden_num, cur_predict_num = \
                    get_ner_fmeasure(decode_result, tag_tensor, my_data.tag_vocab)
                right_num += cur_right_num
                predict_num += cur_predict_num
                golden_num += cur_golden_num
                char_num += cur_char_num
                acc_char_num += cur_acc_num
            except StopIteration:
                data_iter.reset_iter()
                break
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (acc_char_num+0.0)/char_num
    return accuracy, precision, recall, f_measure


def train_model(model, config, args, my_data, use_gpu):
    """
    训练模型 专用
    :param config: 模型的超参数设置
    :param args: 输入的一些保存文件的路径的一些东西
    :param my_data: 里面存放着数据集 vocab、词向量等
    :param use_gpu: 是否使用 GPU
    :return: 不知道返回什么呢哈哈哈啊哈
    """
    best_dev = -1.0  # 最好的F值是什么
    print_interval = 40 # 每十个batch输出一次
    # 准备优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.regularization)
    # 实现的一些功能：lr decay / 模型保存
    # 使用两个作图，一个是 loss 函数的下降图，一个是 f1在dev上面的图
    train_f1_list_with_epoch = []
    loss_list_with_epoch = []
    train_precision_list_with_epoch = []
    train_recall_list_with_epoch = []
    f1_list_with_epoch = []
    recall_list_with_epoch = []
    precision_list_with_epoch = []
    for epoch_idx in range(config.epoch_num):
        print('training epoch:', epoch_idx + 1)
        epoch_start = time.time()
        sample_start = time.time()
        epoch_loss = 0.0
        sample_loss = 0.0
        batch_id = 0
        # 当前batch，我们也要生成其各项指标
        epoch_right_num = 0  # 边界和种类都对的数量累加
        epoch_predict_num = 0  # 预测出的总共的实体数量
        epoch_golden_num = 0  # 真实值中实体的数量
        epoch_char_num = 0  # 总共的字符数
        epoch_acc_char_num = 0  # 标签匹配的字符数
        # 把模型调整到训练时候
        model.train()
        model.zero_grad()
        # 进行学习率衰减
        lr_decay(optimizer, epoch_idx, config.lr_decay, config.lr)
        # 然后我们对epoch内的数据进行随机打乱,不知道行不行
        data_num = len(my_data.train_iter.data_list)
        print("训练数据总量：", data_num)
        # np.random.shuffle(my_data.train_iter.data_list)
        # 下面，如何取得迭代数据，并把其搞到gpu上面
        while True:
            try:
                # 不知道这样子行不行哈哈哈哈哈哈 我真是佛了
                sentence_tensor, tag_tensor, sentence_mask = my_data.train_iter.next()
                # print(sentence_tensor)
                # print(tag_tensor)
                # print(sentence_mask)
                # 转换到gpu上面
                if use_gpu:
                    sentence_tensor = sentence_tensor.cuda()
                    tag_tensor = tag_tensor.cuda()
                    sentence_mask = sentence_mask.cuda()
                #print("句子形状：", sentence_tensor.shape)
                batch_loss_tensor, batch_path = model(sentence_tensor, tag_tensor, sentence_mask, use_gpu)
                epoch_loss += torch.sum(batch_loss_tensor, dim=-1).data
                sample_loss += torch.sum(batch_loss_tensor, dim=-1).data
                batch_loss = torch.mean(batch_loss_tensor, dim=-1)
                # 进行batch_累加：
                cur_acc_num, cur_char_num, cur_right_num, cur_golden_num, cur_predict_num = \
                get_ner_fmeasure(batch_path, tag_tensor, my_data.tag_vocab)
                epoch_right_num += cur_right_num
                epoch_predict_num += cur_predict_num
                epoch_golden_num += cur_golden_num
                epoch_char_num += cur_char_num
                epoch_acc_char_num += cur_acc_num
                model.zero_grad()
                batch_loss.backward()  # 不知道可不可以
                optimizer.step()
                batch_id += 1
                #  print('epoch:{}, batch:{}, loss:{:.4f}'.format(epoch_idx, batch_id, batch_loss.item()))
                if batch_id % print_interval == 0:
                    sample_cost = time.time() - sample_start
                    print("sentence_shape:",sentence_tensor.shape)
                    print('sample loss: epoch:{}, batch:{}, time:{:.2f}s, batch_loss:{:.4f}'.format(epoch_idx+1, batch_id, sample_cost, sample_loss.item()/print_interval))
                    sample_start = time.time()
                    sample_loss = 0.0
            except StopIteration:
                my_data.train_iter.reset_iter()
                break

        if epoch_predict_num == 0:
            epoch_precision = -1
        else:
            epoch_precision = (epoch_right_num+0.0)/epoch_predict_num
        if epoch_golden_num == 0:
            epoch_recall = -1
        else:
            epoch_recall = (epoch_right_num+0.0)/epoch_golden_num
        if (epoch_precision == -1) or (epoch_recall == -1) or (epoch_precision+epoch_recall) <= 0.0:
            epoch_f_measure = -1
        else:
            epoch_f_measure = 2*epoch_precision*epoch_recall/(epoch_precision+epoch_recall)
        epoch_accuracy = (epoch_acc_char_num+0.0)/epoch_char_num
        epoch_end = time.time()
        epoch_cost = epoch_end - epoch_start
        print('Epoch:{} training finished. Time: {:.2f}s, '
              'speed: {:.2f}s/instance, total loss:{:.4f}'.format(epoch_idx + 1,epoch_cost,epoch_cost/data_num,epoch_loss.item()))
        print('Epoch_train_result: epoch:{}, accuracy:{:.4f}, '
              'precision:{:.4f}, recall:{:.4f}, f_measure:{:.4f} '.format(epoch_idx+1, epoch_accuracy, epoch_precision, epoch_recall, epoch_f_measure))
        acc, p, r, f = eval_model(model, my_data, use_gpu=use_gpu)
        dev_cost = time.time() - epoch_end
        print('dev_result: epoch:{}, time:{:.2f}, accuracy:{:.4f}, '
              'precision:{:.4f}, recall:{:.4f}, f_measure:{:.4f} '.format(epoch_idx+1, dev_cost, acc, p, r, f))
        # 一系列累加 哇哈哈哈哈哈哈
        loss_list_with_epoch.append(epoch_loss.item())
        train_f1_list_with_epoch.append(epoch_f_measure)
        train_precision_list_with_epoch.append(epoch_precision)
        train_recall_list_with_epoch.append(epoch_recall)
        f1_list_with_epoch.append(f)
        recall_list_with_epoch.append(r)
        precision_list_with_epoch.append(p)
        if f > best_dev:
            print("epoch:{} dev set exceed best f score{:.4f}".format(epoch_idx+1, best_dev))
            best_dev = f
            # 如果要运行多个模型，这个就需要注意了哦！
            # save_model 应该是一个包含了模型名称的路径
            model_name = args.save_model + os.sep + 'epoch_' + str(epoch_idx)+ '.model'
            torch.save(model.state_dict(), model_name)
    print("loss list:",loss_list_with_epoch)
    print('train f1 list:',train_f1_list_with_epoch)
    print('train precision list:',train_precision_list_with_epoch)
    print('train recall list:',train_recall_list_with_epoch)
    print("dev f1 list:",f1_list_with_epoch)
    print("dev precision list:",precision_list_with_epoch)
    print("dev recall list:",recall_list_with_epoch)   

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description='tuning and setting the various Transformer')
    opt.add_argument('--model',choices=['xl','reformer'],default='xl')
    opt.add_argument('--status', choices=['train', 'test', 'decode'], default='train')
    opt.add_argument('--save_model', default='data/Ontonotes_5.0/saved_model/xl')
    opt.add_argument('--load_model', help='training from scratch or load path', default=None)
    opt.add_argument('--save_data', help='save vocab or vector', default=None)
    opt.add_argument('--load_data', help='load vocab or vector', default=None)
    opt.add_argument('--train', help='the path of train file', default='data/Ontonotes_5.0/ontonotes_train_bioes')
    opt.add_argument('--dev', help='the path of dev file', default='data/Ontonotes_5.0/ontonotes_dev_bioes')
    opt.add_argument('--test', help='the path of test file', default='data/Ontonotes_5.0/ontonotes_test_bioes')
    args = opt.parse_args()
    # gpu 是否可以使用
    use_gpu = torch.cuda.is_available()
    print("gpu是否空闲：", use_gpu)
    # 将status变为小写
    status = args.status.lower()
    # 刷新缓冲区
    sys.stdout.flush()
    # 下面 我们初始化超参数以及输入
    # 将其放入一个文件中，统一进行配置
    if args.model == 'xl':
        config = xl_config()
    # elif args.model == 'universal':
    #     config = universal_config()
    data = my_data.Data(args)
    # 经过如下的build_data,我们已经初始化了字词向量
    data.build_data()
    # 对于下面的几句话，不管是训练还是测试，都是一样的。
    src_embedding_num = data.char_vocab.get_length()
    tag_num = data.tag_vocab.get_length() - 1
    # 得到模型，这块，要根据究竟选择了哪些模型进行得到模型
    #print('加载模型前当前进程的内存使用：{:.4f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
    model = get_model(config, src_embedding_num, tag_num, data.char_embedding, data.char_embedding_dim, use_gpu, args.load_model)
    # 下面 我们进行判断，进行模型训练
    #print('加载模型后当前进程的内存使用：{:.4f} GB'.format((psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024)))
    if status == 'train':
        train_model(model, config, args, data, use_gpu)
    elif status == 'test':
        start_time = time.time()
        print(model.config.reuse_seg_num)
        print(model.encoder.relative_encoding.shape)
        a,p,r,f= eval_model(model, data, mode='test', use_gpu=use_gpu)
        print('test_result:  time:{:.2f}, accuracy:{:.4f}, '
             'precision:{:.4f}, recall:{:.4f}, f_measure:{:.4f} '.format(time.time()-start_time, a, p, r, f))
    elif status == 'decode':
        # 输入一个句子，我们进行decode,所以当我们decode的时候，在命令行一定要load_model
        decode(model, data, use_gpu)
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
