import os
import codecs
dir_name= '/home/wqt/long_char_model/data/Ontonotes_5.0'
left=['PERSON','GPE','LOC','ORG']
MAX_LEN = 2000
# for root_,dirs,files in os.walk(dir_name):
#     for file in files:
file = 'ontonotes_dev.bioes'
num = 0
with codecs.open(dir_name + os.sep + file,'r','utf-8') as fin:
    with codecs.open(dir_name +os.sep+ file + '_','w','utf-8') as fout:
        after_process_seg_num = 0
        seg_word_list = []
        seg_tag_list = []
        seg_char_num = 0
        sent_word_list =[]
        sent_tag_list = []
        is_before_seg_part = False
        for line in fin.readlines():
            if line not in['\n', '\r\n'] and not(line.strip().startswith("#") and line.strip().endswith("#")):
                word_label = line.strip().split()
                if len(word_label) >= 2:
                    word = word_label[0]
                    label = word_label[1]
                    if label[2:] not in left:
                        label='O'
                    seg_char_num += 1
                    sent_word_list.append(word)
                    sent_tag_list.append(label)
                is_before_seg_part = False

            if line in ['\n', '\r\n']:
                    # 如果当前行是换行的话
                seg_word_list.append(sent_word_list)
                seg_tag_list.append(sent_tag_list)
                sent_word_list = []
                sent_tag_list = []
                is_before_seg_part = False
            
            if line.strip().startswith("#") and line.strip().endswith("#") and not is_before_seg_part:
                # 当前 seg 太长了，
                is_before_seg_part = True
                # 第一个要分段的index
                end_pos = len(seg_word_list) - 1
                interval = 0
                if seg_char_num > MAX_LEN:
                    if seg_char_num <= 2 * MAX_LEN:
                        interval = len(seg_word_list)//2
                        after_process_seg_num += 2
                    else:
                        interval = len(seg_word_list) // 3
                        after_process_seg_num += 3
                    # 如果要分开 数据，要分段的 index
                    end_pos = 0 + interval
                if seg_char_num <= MAX_LEN:
                    after_process_seg_num += 1  
                for i,sent in enumerate(seg_word_list):
                    sentence = seg_word_list[i]
                    tag = seg_tag_list[i]
                    for j in range(len(sentence)):
                        fout.write('\t'.join([sentence[j],tag[j]]))
                        fout.write('\n')
                    # 写一个换行
                    fout.write('\n')
                    if i == end_pos:
                        num += 1
                        fout.write("##################################")
                        fout.write("\n")
                        # 进行最后一次判断的时候，会产生一系列碎片。
                        end_pos = min(end_pos + interval, len(seg_word_list)-1)
                        if len(seg_word_list)-1-end_pos<=5:
                            end_pos = len(seg_word_list)-1
                seg_word_list = []
                seg_tag_list = []
                seg_char_num = 0
                sent_word_list = []
                sent_tag_list = []
        print(file, after_process_seg_num)
        print(num)
