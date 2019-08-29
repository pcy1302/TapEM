import pickle as pkl
import time
import pdb
import numpy as np
import torch
from Dataset import Dataset
from evaluate import Evaluator
gpu_dict = {5: 1, 6: 2, 7: 3, 4: 0, 0: 4, 1: 5, 2: 6, 3: 7, -1: 'cpu'}
class embedder:
    def __init__(self, args):
        # parameters setting
        self.top_K = [int(elem) for elem in eval(args.top_K)]
        self.embedder = args.embedder
        self.embed_d = args.embed_dim
        self.hidden_n = args.embed_dim
        self.c_len = args.c_len
        self.c_reg = args.c_reg
        self.margin_d = args.margin_d
        self.c_tradeoff = args.c_tradeoff
        self.batch_s = args.batch_size
        self.lr = args.learn_rate
        self.iter_max = args.train_iter_max
        self.data_path = args.data_path
        self.version = args.version
        self.year = args.year

        self.model_path = args.model_path
        self.gpu_num = gpu_dict[args.gpu_num]

        self.device = torch.device("cuda:" + str(self.gpu_num) if torch.cuda.is_available() else "cpu")

        self.input_data = Dataset(args)
        self.word_embed = self.input_data.word_embed
        self.word_n = self.word_embed.shape[0] - 2
        self.word_dim = self.word_embed.shape[1]

        self.dnn_dims = [int(elem) for elem in eval(args.dnn_dims)]
        self.num_ctx_neg = args.num_ctx_neg
        self.metric = args.metric
        self.early_stop = args.early_stop
        self.save = args.save
        self.init_std = args.init_std
        self.reg_ml = args.reg_ml
        self.reg_semi = args.reg_semi
        self.reg_ctx = args.reg_ctx
        self.ctx_window = args.ctx_window
        self.pretrain = args.pretrain
        self.rescale_grad = args.rescale_grad
        self.grad_norm = args.grad_norm
        self.ctx_ratio = args.ctx_ratio
        self.scheduler_factor = args.scheduler_factor
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_threshold = args.scheduler_threshold
        self.pretrained_f_name = "_{}_{}".format(args.version, args.year)
        self.evaluator = Evaluator(self.input_data, self.metric, self.early_stop, self.top_K, self.save, args)

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
