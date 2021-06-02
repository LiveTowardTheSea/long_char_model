path = '/home/wqt/long_char_model/data/Ontonotes_5.0/ontonotes_test_bioes'
import codecs 
len_list = [0]* 20
para_len = 0
with codecs.open(path,'r','utf-8')as f:
    for line in f.readlines():
        if line.strip().startswith("#") and line.strip().endswith("#"):
            len_list[para_len//200] += 1
            para_len = 0
        elif line in ['\n', '\r\n']:
            pass
        elif len(line) >= 2:
            para_len += 1
    print(len_list)

