import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import random
from embedder import embedder
import sys

np.random.seed(0)
random.seed(0)

class camel(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)

    def pad(self, list_of_tensors):
        list_of_tensors = [torch.LongTensor(elem[:self.c_len]) for elem in list_of_tensors]
        seq_lengths = torch.LongTensor([len(elem) for elem in list_of_tensors])
        seq_tensor = pad_sequence(list_of_tensors, batch_first=True)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, perm_idx, seq_lengths

    def training(self):
        model = modeler(self.input_data, self.embed_d, self.word_n, self.word_dim, self.device).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)

        for epoch in range(1, self.iter_max):
            totalLoss = 0

            p_a_a_dir_batch = self.input_data.p_a_a_dir_next_batch_negsample().to(self.device)
            p_c_dir_batch = self.input_data.gen_content_batch(p_a_a_dir_batch)
            p_c_dir_batch, perm_idx_dir, seq_lengths_dir = self.pad(p_c_dir_batch)
            p_a_a_dir_batch = p_a_a_dir_batch[perm_idx_dir]
            p_c_dir_batch = p_c_dir_batch.to(self.device)

            p_a_a_indir_batch = self.input_data.p_a_a_indir_next_batch().to(self.device)
            p_c_indir_batch = self.input_data.gen_content_batch(p_a_a_indir_batch)
            p_c_indir_batch, perm_idx_indir, seq_lengths_indir = self.pad(p_c_indir_batch)
            p_a_a_indir_batch = p_a_a_indir_batch[perm_idx_indir]
            p_c_indir_batch = p_c_indir_batch.to(self.device)
            mini_batch_n = int(len(p_a_a_dir_batch) / self.batch_s)
            for i in range(mini_batch_n):
                p_a_a_dir_mini_batch = p_a_a_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                p_c_dir_mini_batch = p_c_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                seq_lengths_dir_mini_batch = seq_lengths_dir[i * self.batch_s:(i + 1) * self.batch_s]

                p_a_a_indir_mini_batch = p_a_a_indir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                p_c_indir_mini_batch = p_c_indir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                seq_lengths_indir_mini_batch = seq_lengths_indir[i * self.batch_s:(i + 1) * self.batch_s]

                optimizer.zero_grad()

                pos_dir, neg_dir, sum1, sum2 = model(p_a_a_dir_mini_batch, p_a_a_indir_mini_batch, p_c_dir_mini_batch,
                                                     p_c_indir_mini_batch, seq_lengths_dir_mini_batch,
                                                     seq_lengths_indir_mini_batch, isTrain=True)

                Loss_1 = torch.sum(torch.max(pos_dir - neg_dir + self.margin_d, torch.Tensor([0]).to(self.device)))
                Loss_2 = (-(sum1 + sum2)).sum()

                reg_loss = None
                for param in list(filter(lambda p: p.requires_grad, model.parameters())):
                    if reg_loss is None:
                        reg_loss = (param ** 2).sum()
                    else:
                        reg_loss = reg_loss + (param ** 2).sum()

                loss = Loss_1 + self.c_tradeoff * Loss_2 + self.c_reg * reg_loss

                loss.backward()
                optimizer.step()

                totalLoss += loss.item()
                del loss


            st = "[{}][{}][Iter {:3}] loss: {:3}".format(self.currentTime(), self.embedder, epoch, round(totalLoss,2))
            model.eval()
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            torch.save(model.state_dict(), self.model_path + "/camel{}.pth".format(self.pretrained_f_name))
            p_text_all = [self.input_data.p_content[i] for i in self.input_data.test_p_id_list]

            p_text_all, perm_idx, seq_lengths = self.pad(p_text_all)
            p_text_deep_f = model([], [], p_text_all, [], seq_lengths, [], isTrain=False)
            perm_idx = perm_idx.numpy().argsort()
            p_text_deep_f = p_text_deep_f[perm_idx]

            p_text_deep_f = p_text_deep_f.detach().cpu().numpy()
            a_latent_f = model.author_embed.weight.data.detach().cpu().numpy()

            is_converged = self.evaluator.evaluate_Camel(model, self.embedder, self.model_path, st, epoch, p_text_deep_f, a_latent_f)
            if is_converged:
                print("Converged!")
                return
            model.train()



class modeler(nn.Module):
    def __init__(self, input_data, embed_d, word_n, word_dim, device):
        super(modeler, self).__init__()
        self.author_embed = nn.Embedding(input_data.author_num, embed_d)
        self.word_embed = nn.Embedding(word_n + 2, word_dim)
        self.word_embed.weight.requires_grad = False
        self.rnn_words = nn.GRU(word_dim, embed_d, batch_first=True)
        self.bias = nn.Parameter(torch.zeros((1,1)))
        self.logsigmoid = nn.LogSigmoid()
        self.init_weights(input_data.word_embed)

        self.embed_d = embed_d
        self.input_data = input_data
        self.device = device

    def init_weights(self, word_embed):
        nn.init.normal_(self.author_embed.weight.data, mean=0.0, std=0.01)
        self.bias.data.fill_(0.1)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

    def forward(self, p_a_a_dir, p_a_a_indir, p_c_dir_input, p_c_indir_input, seq_lengths_dir, seq_lengths_indir, isTrain=True):
        if isTrain:
            # Metric learning loss
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
            packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
            p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float().to(self.device)

        if not isTrain:
            # Metric learning loss
            self.word_embed = self.word_embed.cpu()
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
            packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)
            self.rnn_words = self.rnn_words.cpu()
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
            p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float()
            self.word_embed = self.word_embed.to(self.device)
            self.rnn_words = self.rnn_words.to(self.device)
            return p_c_dir_e

        a_e_pos = self.author_embed(p_a_a_dir[:, 1])
        a_e_neg = self.author_embed(p_a_a_dir[:, 2])

        pos_dir = torch.sum((p_c_dir_e - a_e_pos) ** 2, 1)
        neg_dir = torch.sum((p_c_dir_e - a_e_neg) ** 2, 1)

        # Random walk loss
        p_c_indir_word_e = self.word_embed(p_c_indir_input)
        packed_input = pack_padded_sequence(p_c_indir_word_e, seq_lengths_indir.cpu().numpy(), batch_first=True)
        p_c_indir_deep_e, _ = self.rnn_words(packed_input)
        p_c_indir_deep_e, _ = pad_packed_sequence(p_c_indir_deep_e, batch_first=True)

        p_c_indir_e = torch.sum(p_c_indir_deep_e, 1) / seq_lengths_indir.unsqueeze(1).float().to(self.device)
        a_e_pos = self.author_embed(p_a_a_indir[:, 1])
        a_e_neg = self.author_embed(p_a_a_indir[:, 2])

        sum1 = torch.sum(p_c_indir_e * a_e_pos, 1) + self.bias
        sum2 = torch.sum(p_c_indir_e * a_e_neg, 1) + self.bias
        sum1 = self.logsigmoid(sum1)
        sum2 = self.logsigmoid(-sum2)

        return pos_dir, neg_dir, sum1, sum2
