class xl_config:
    def __init__(self):
        self.model_name = 'xl'
        self.d_model = 256
        self.head = 4
        self.k_dim = 64
        self.d_ff = 1024
        self.first_layer_num = 3
        self.second_layer_num = 1
        self.seg_len = 128
        self.reuse_seg_num = 4
        self.p_drop=0.1
        self.regularization = 0.02
        self.lr = 0.001
        self.lr_decay = 0.01
        self.epoch_num = 30