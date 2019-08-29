import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import random
import torch.nn.functional as F
from embedder import embedder
import sys
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

np.random.seed(0)
random.seed(0)

class TapEM(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def pad(self, list_of_tensors):
        list_of_tensors = [torch.LongTensor(elem[:self.c_len]) for elem in list_of_tensors]
        seq_lengths = torch.LongTensor([len(elem) for elem in list_of_tensors])
        seq_tensor = pad_sequence(list_of_tensors, batch_first=True)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, perm_idx, seq_lengths

    def rescale_gradients(self, model, grad_norm):
        parameters_to_clip = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        nn.utils.clip_grad_norm_(parameters_to_clip, grad_norm)

    def training(self):
        model_attention = AttentivePooling(self.dnn_dims)
        model = modeler(self.args, self.input_data, model_attention, self.input_data.author_num, self.embed_d,
                        self.word_n, self.word_dim, self.dnn_dims, self.c_len, self.init_std,
                        self.model_path+"/camel"+self.pretrained_f_name+".pth", self.device, self.pretrain).to(self.device)

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = optim.Adam(parameters, lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.scheduler_factor,
                                      patience=self.scheduler_patience, verbose=True, threshold=self.scheduler_threshold)
        for epoch in range(self.iter_max):
            torch.cuda.empty_cache()
            totalLoss = 0

            # Prepare data for metric learning and pair validity classifier
            p_a_a_dir_batch = self.input_data.p_a_a_dir_next_batch_negsample().to(self.device)
            p_c_dir_batch = self.input_data.gen_content_batch(p_a_a_dir_batch)
            p_c_dir_batch, perm_idx_dir, seq_lengths_dir = self.pad(p_c_dir_batch)
            p_a_a_dir_batch = p_a_a_dir_batch[perm_idx_dir]
            p_c_dir_batch = p_c_dir_batch.to(self.device)

            # Prepare data for pair context
            ctx_lens = list(self.input_data.pair_context_label[0].keys())
            curr_ctx = ctx_lens[epoch % len(ctx_lens)]
            pair_context_total_pair = self.input_data.pair_context_label[0][curr_ctx]
            pair_context_total_context = self.input_data.pair_context_label[1][curr_ctx]

            # randomly sample pair context (b/c too many samples)
            random_sample_idxs = random.sample(range(len(pair_context_total_pair)), int(len(pair_context_total_pair) * self.ctx_ratio))
            pair_context_batch_pair = [pair_context_total_pair[idx] for idx in random_sample_idxs]
            pair_context_batch_context = [pair_context_total_context[idx] for idx in random_sample_idxs]

            mini_batch_n = int(len(p_a_a_dir_batch) / self.batch_s)
            batch_s_ctx = int(len(pair_context_batch_pair) / mini_batch_n)

            for i in range(mini_batch_n):
                # metric learning and pair validity classifier
                p_a_a_dir_mini_batch = p_a_a_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                p_c_dir_mini_batch = p_c_dir_batch[i * self.batch_s:(i + 1) * self.batch_s]
                seq_lengths_dir_mini_batch = seq_lengths_dir[i * self.batch_s:(i + 1) * self.batch_s]

                # pair context
                pair_context_mini_batch_pair = pair_context_batch_pair[i * batch_s_ctx:(i + 1) * batch_s_ctx]
                pair_context_mini_batch_context = pair_context_batch_context[i * batch_s_ctx:(i + 1) * batch_s_ctx]

                negative_contexts_mini_batch = []
                for _ in range(self.num_ctx_neg):
                    for idx, (pair, context) in enumerate(zip(pair_context_mini_batch_pair, pair_context_mini_batch_context)):
                        neg_ctx_idx = random.randint(0, len(pair_context_batch_context) - 1)
                        context_neg = pair_context_batch_context[neg_ctx_idx]
                        while (len(context_neg) != len(context)) or (type(context[0]) != type(context_neg[0])) or (context_neg == context):
                            neg_ctx_idx = random.randint(0, len(pair_context_batch_context) - 1)
                            context_neg = pair_context_batch_context[neg_ctx_idx]
                        negative_contexts_mini_batch.append(context_neg)

                for k in range(self.num_ctx_neg):
                    pair_context_mini_batch_pair += pair_context_batch_pair[i * batch_s_ctx:(i + 1) * batch_s_ctx]
                    pair_context_mini_batch_context += negative_contexts_mini_batch[k * batch_s_ctx: (k + 1) * batch_s_ctx]

                pos_dir_loss, neg_dir_loss, pos_semi_loss, neg_semi_loss, pos_ctx_loss, neg_ctx_loss \
                    = model(p_a_a_dir_mini_batch, p_c_dir_mini_batch, seq_lengths_dir_mini_batch,
                            pair_context_mini_batch_pair, pair_context_mini_batch_context, len(negative_contexts_mini_batch))


                optimizer.zero_grad()

                loss1 = torch.sum(torch.max(pos_dir_loss - neg_dir_loss + self.margin_d, torch.Tensor([0]).to(self.device)))
                loss2 = (-(pos_semi_loss + neg_semi_loss))
                loss3 = (-(pos_ctx_loss + neg_ctx_loss))

                loss = loss1 + loss2 + loss3

                loss.backward()
                if self.rescale_grad:
                    self.rescale_gradients(model, self.grad_norm)
                optimizer.step()
                totalLoss += loss.item()
                del loss

            del p_a_a_dir_batch

            st = "[{}][{}][Iter {:3}] loss: {:6}".format(self.currentTime(), self.embedder, epoch, round(totalLoss, 2))

            model.eval()
            torch.cuda.empty_cache()
            is_converged = self.evaluator.evaluate_TapEM(self.embedder, self.model_path, st, epoch, model)
            model.final_output = model.final_output.to(self.device)
            curr_recall = self.evaluator.recall_ave_dev_dict_list[2][-1]
            scheduler.step(curr_recall)

            if is_converged:
                print("Converged!")
                return
            model.train()
            sys.stdout.flush()
            print("[{}]Done Epoch {}".format(self.currentTime(), epoch))


class modeler(nn.Module):
    def __init__(self, args, input_data, model_attention, author_num, embed_d, word_n, word_dim, dnn_dims,
                 c_len, init_std, pretrain_path, device, pretrain=True):
        super(modeler, self).__init__()
        self.model_attention = model_attention
        self.author_embed = nn.Embedding(author_num, embed_d)
        self.word_embed = nn.Embedding(word_n + 2, word_dim)
        self.word_embed.weight.requires_grad = False
        self.rnn_words = nn.GRU(word_dim, embed_d, batch_first=True)
        self.rnn_contexts = nn.GRU(embed_d, dnn_dims[-1], batch_first=True, bidirectional=True)
        self.project_output = nn.Linear(dnn_dims[-1] * 2, dnn_dims[-1])
        self.bias = nn.Parameter(torch.zeros((1, 1)))
        self.logsigmoid = nn.LogSigmoid()
        self.dropout = nn.Dropout(p=args.dropout)
        fc_modules = []
        f_in = embed_d * 4
        for i, f_out in enumerate(dnn_dims):
            if i < len(dnn_dims) - 1:
                fc_modules.append(self.dropout)
                fc_modules.append(nn.Linear(f_in, f_out))
                fc_modules.append(nn.LayerNorm(f_out))
                fc_modules.append(nn.ReLU())
                f_in = f_out
            else:  # last layer
                fc_modules.append(self.dropout)
                fc_modules.append(nn.Linear(f_in, f_out))
                fc_modules.append(nn.LayerNorm(f_out))
        self.edge_MLP = nn.Sequential(*fc_modules).to(device)

        self.embed_d = embed_d
        self.input_data = input_data
        self.c_len = c_len
        self.init_std = init_std
        self.device = device
        self.dnn_dims = dnn_dims
        self.pretrain_path = pretrain_path
        self.pretrain = pretrain

        fc_modules = []
        fc_modules.append(nn.Linear(dnn_dims[-1], int(dnn_dims[-1])))
        fc_modules.append(nn.ReLU())
        fc_modules.append(nn.Linear(int(dnn_dims[-1]), 1))
        self.final_output = nn.Sequential(*fc_modules).to(device)
        self.init_weights(input_data.word_embed)

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s

    def init_weights(self, word_embed):
        if self.pretrain:
            print("Loading pretrained weights from {}".format(self.pretrain_path))
            pretrained_dict = torch.load(self.pretrain_path, map_location=self.device)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            nn.init.normal_(self.author_embed.weight.data, mean=0.0, std=self.init_std)
            self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

        for i in range(len(self.edge_MLP)):
            if i % 4 == 1:
                nn.init.normal_(self.edge_MLP[i].weight.data, mean=0.0, std=self.init_std)
                # self.edge_MLP[i].bias.data.fill_(0.1)

        nn.init.normal_(self.final_output[0].weight.data, mean=0.0, std=self.init_std)
        nn.init.normal_(self.final_output[2].weight.data, mean=0.0, std=self.init_std)
        # self.final_output[0].bias.data.fill_(0.1)
        # self.final_output[2].bias.data.fill_(0.1)

        nn.init.normal_(self.project_output.weight.data, mean=0.0, std=0.01)

    def pad(self, list_of_tensors):
        list_of_tensors = [torch.LongTensor(elem[:self.c_len]) for elem in list_of_tensors]
        seq_lengths = torch.LongTensor([len(elem) for elem in list_of_tensors])
        seq_tensor = pad_sequence(list_of_tensors, batch_first=True)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        return seq_tensor, perm_idx, seq_lengths

    def get_word_embed_dir(self, p_c_dir_input, seq_lengths_dir, isEval=False):
        if not isEval:
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
        else:
            self.word_embed = self.word_embed.cpu()
            p_c_dir_word_e = self.word_embed(p_c_dir_input)
            self.word_embed = self.word_embed.to(self.device)
        packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)

        if not isEval:
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
        else:
            self.rnn_words = self.rnn_words.cpu()
            p_c_dir_deep_e, _ = self.rnn_words(packed_input)
            self.rnn_words = self.rnn_words.to(self.device)

        p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

        if not isEval:
            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float().to(self.device)
        else:
            p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float()



        del p_c_dir_input
        del packed_input

        return p_c_dir_e

    def make_input(self, left, right):
        output = torch.cat((left, right, left * right, left - right), dim=1)

        return output

    # get representation of papers after rnn layers
    def get_representation(self, paper_content, isEval=False):
        p_c_all, perm_idx_all, seq_lengths_all = self.pad(paper_content)
        if not isEval:
            p_c_all = p_c_all.to(self.device)
        p_c_all_e = self.get_word_embed_dir(p_c_all, seq_lengths_all, isEval)
        # back to original order
        p_c_all_e = p_c_all_e[perm_idx_all.numpy().argsort()]

        del p_c_all
        return p_c_all_e

    def store_rnn_matrix(self, papers, isEval=False):
        paper_content = self.input_data.gen_content(papers)
        paper_content_vecs = self.get_representation(paper_content, isEval)

        return paper_content_vecs


    def get_author_embed(self, authors_idxs):

        return self.author_embed(torch.LongTensor(authors_idxs).to(self.device)) # authors

    def concat(self, authors, papers, isCtx, isEval=False):
        combined = self.make_input(authors, papers)
        if not isEval:
            R_x_y = self.edge_MLP(combined)
        else:
            self.edge_MLP = self.edge_MLP.cpu()
            R_x_y = self.edge_MLP(combined)
            self.edge_MLP = self.edge_MLP.to(self.device)

        if isCtx:
            return R_x_y

        combined = self.make_input(papers, authors)
        R_y_x = self.edge_MLP(combined)

        R = torch.cat((R_x_y, R_y_x), 1)

        return R


    def get_pair_edge(self, author_idxs, paper_idxs, isPair):
        paper_list = sorted(paper_idxs)

        paper_map = {}
        paper_matrix = self.store_rnn_matrix(paper_list)
        for idx, p_id in enumerate(paper_list):
            paper_map[p_id] = idx

        author_embeds = self.get_author_embed(author_idxs)
        paper_mapped_idxs = [paper_map[idx] for idx in paper_idxs]
        paper_embeds = paper_matrix[paper_mapped_idxs]

        if isPair:
            return self.concat(author_embeds, paper_embeds, isCtx=True)
        else:
            return author_embeds, paper_embeds

    def forward(self, p_a_a_dir, p_c_dir_input, seq_lengths_dir, pair_context_pair, pair_context_context, num_neg_ctx):

        # Metric learning
        p_c_dir_word_e = self.word_embed(p_c_dir_input)
        packed_input = pack_padded_sequence(p_c_dir_word_e, seq_lengths_dir.cpu().numpy(), batch_first=True)
        p_c_dir_deep_e, _ = self.rnn_words(packed_input)
        p_c_dir_deep_e, _ = pad_packed_sequence(p_c_dir_deep_e, batch_first=True)

        p_c_dir_e = torch.sum(p_c_dir_deep_e, 1) / seq_lengths_dir.unsqueeze(1).float().to(self.device)

        a_e_pos = self.author_embed(p_a_a_dir[:, 1])
        a_e_neg = self.author_embed(p_a_a_dir[:, 2])

        pos_dir_loss = torch.sum((p_c_dir_e - a_e_pos) ** 2, 1)
        neg_dir_loss = torch.sum((p_c_dir_e - a_e_neg) ** 2, 1)

        # Semi
        paper_list = p_a_a_dir[:, 0].cpu().numpy()
        authors_pos = p_a_a_dir[:, 1].cpu().numpy()
        authors_neg = p_a_a_dir[:, 2].cpu().numpy()

        pos_edges = self.get_pair_edge(authors_pos, paper_list, isPair=True)
        neg_edges = self.get_pair_edge(authors_neg, paper_list, isPair=True)

        pos_output = self.final_output(pos_edges)
        neg_output = self.final_output(neg_edges)

        pos_semi_loss = self.logsigmoid(pos_output).sum()
        neg_semi_loss = self.logsigmoid(-neg_output).sum()

        # For Pair
        author_idxs = [elem[0][1] for elem in pair_context_pair]
        paper_idxs = [elem[1][0] for elem in pair_context_pair]
        R_x_y = self.get_pair_edge(author_idxs, paper_idxs, isPair=True)

        # For context
        author_idxs = [elem[1] for context in pair_context_context for elem in context if len(elem) == 2]
        paper_idxs = [elem[0] for context in pair_context_context for elem in context if len(elem) == 1]

        # paper_list = set([elem for a_id in author_idxs for elem in self.input_data.a_idx_p_dir_dict_train[a_id]])
        # paper_list = sorted(list(paper_list.union(set(list(paper_idxs)))))
        paper_list = paper_idxs

        paper_map = {}
        paper_matrix = self.store_rnn_matrix(paper_list)
        for idx, p_id in enumerate(paper_list):
            paper_map[p_id] = idx

        author_embeds = self.get_author_embed(author_idxs)
        paper_mapped_idxs = [paper_map[idx] for idx in paper_idxs]
        paper_embeds = paper_matrix[paper_mapped_idxs]

        contexts = torch.stack((author_embeds, paper_embeds), dim=1).view(-1, len(pair_context_context[0]), self.embed_d)

        # del paper_matrix
        # del author_embeds
        # del paper_embeds

        output, _ = self.rnn_contexts(contexts)
        output_context = self.project_output(output)
        output_context = self.model_attention(output_context, self.device)

        prob = torch.sum(R_x_y * output_context, 1)

        pos_ctx_loss = prob[:len(prob) - num_neg_ctx]
        neg_ctx_loss = prob[len(prob)-num_neg_ctx:]

        pos_ctx_loss = self.logsigmoid(pos_ctx_loss).sum()
        neg_ctx_loss = self.logsigmoid(-neg_ctx_loss).sum()

        return pos_dir_loss, neg_dir_loss, pos_semi_loss, neg_semi_loss, pos_ctx_loss, neg_ctx_loss


class AttentivePooling(nn.Module):
    def __init__(self, dnn_dims):
        super(AttentivePooling, self).__init__()
        self.dnn_dims = dnn_dims
        self.k = nn.Linear(self.dnn_dims[-1], 1)
        self.W = nn.Parameter(torch.zeros(self.dnn_dims[-1], self.dnn_dims[-1]))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.k.weight.data, mean=0.0, std=0.01)
        nn.init.normal_(self.W, mean=0.0, std=0.01)

    def forward(self, output, device):
        attns = self.k(output).squeeze(-1)
        attns = F.softmax(attns, 1)
        output = output.matmul(self.W)
        attns = attns.unsqueeze(1)

        output_context = torch.bmm(attns, output).squeeze(1)

        return output_context
