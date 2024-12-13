import warnings


class DefaultConfig(object):
    load_img_path = 'image_model.pth'  # load model path
    load_txt_path = 'text_model.pth'

    # data parameters
    pretrain_model_path = './data/imagenet-vgg-f.mat'

    # hyper-parameters
    batch_size = 32
    max_epoch = 400
    gamma = 1
    eta = 1
    bit = 32  # final binary code length
    lr = 1e-2 #10 ** (-1.5)  initial learning rate
    use_gpu = True
    device = 'cuda:0'
    valid = True
    print_freq = 2  # print info every N epoch
    # result_dir = 'result'
    seed = 110
    # ----------------------
    group = 'StegaStamp_test'
    # backdoor
    backdoor_trigger = 'StegaStamp'  # BadNets, Blended, WaNet, StegaStamp

    # backdoor hyper-parameters
    pr = 0.1  # poison ration
    backdoor = False  # backdoor for label attack
    backdoor_loss = False  # bakcdoor for loss attack
    poison_train_print = True
    # -----------------------
    # backdoor_loss
    lg = 0.0
    # data parameters
    mapi2t_min = mapt2i_min = 0

    # defense
    denoise_type = 'No'
    prune_rate = 0
    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = './data/FLICKR-25K.mat'
            self.database_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 5000
            self.backdoor_model = 'white'
        if flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE.mat'
            self.database_size = 193734
            self.num_label = 21
            self.query_size = 2000
            self.text_dim = 1000
            self.training_size = 5000
            self.backdoor_model = 'white'
            self.mapi2t_min = 0.64
            self.mapt2i_min = 0.64
        if flag == 'fvc':
            self.dataset = 'fashionvc'
            self.data_path = './data/FashionVC'
            self.database_size = 19862
            self.num_label = 35
            self.query_size = 3000
            self.text_dim = 2685
            self.training_size = 5000
            self.backdoor_model = 'black'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
        self.path = 'checkpoints/' + self.dataset + '/' + str(self.bit) + '/' + self.group + '/'
        # print(self.path)
        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = DefaultConfig()
