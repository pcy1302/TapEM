import six.moves.cPickle as pickle
import numpy as np
import re
import random
import torch
from random import shuffle

np.random.seed(0)
random.seed(0)

class Dataset:
    def __init__(self, args):
        self.args = args
        self.data_path = self.args.data_path
        # direct paper-author relation
        # authors are before conversion
        self.p_a_dir_dict_train = {}
        self.p_a_dir_dict_test = {}
        self.author_train_map = {}
        self.file_name = "_{}_{}".format(args.version, args.year)
        print("Creating direct triples")
        # train should come first in the list
        idx = 0
        dir_relation_f = ["/paper-author-list-train" + self.file_name + ".txt",
                          "/paper-author-list-test" + self.file_name + ".txt"]
        for f_name in dir_relation_f:
            neigh_f = open(self.data_path + f_name, "r")
            print("Reading {}".format(self.data_path + f_name))
            lines = neigh_f.readlines()
            for line in lines:
                line = line.strip()
                p_index = int(re.split(':', line)[0])
                a_list = re.split(',', re.split(':', line)[1])
                if f_name == "/paper-author-list-train" + self.file_name + ".txt":
                    for author in a_list:
                        self.p_a_dir_dict_train.setdefault(p_index, []).append(int(author))
                        if int(author) not in self.author_train_map:
                            self.author_train_map[int(author)] = idx
                            idx += 1
                elif f_name == "/paper-author-list-test" + self.file_name + ".txt":
                    for author in a_list:
                        if int(author) in self.author_train_map:
                            self.p_a_dir_dict_test.setdefault(p_index, []).append(int(author))
            neigh_f.close()

        self.dir_len = sum([len(self.p_a_dir_dict_train[x]) for x in self.p_a_dir_dict_train])

        # get direct author-paper relation
        self.a_p_dir_dict_train = {}
        self.a_idx_p_dir_dict_train = {}
        for p_id, a_ids in self.p_a_dir_dict_train.items():
            for a_id in a_ids:
                self.a_p_dir_dict_train.setdefault(a_id, []).append(p_id)
                self.a_idx_p_dir_dict_train.setdefault(self.author_train_map[a_id], []).append(p_id)

        self.a_p_dir_dict_test = {}
        self.a_idx_p_dir_dict_test = {}
        for p_id, a_ids in self.p_a_dir_dict_test.items():
            for a_id in a_ids:
                self.a_p_dir_dict_test.setdefault(a_id, []).append(p_id)
                self.a_idx_p_dir_dict_test.setdefault(self.author_train_map[a_id], []).append(p_id)

        print("Done creating direct triples")
        self.a_id_list = sorted(list(set(self.a_idx_p_dir_dict_train.keys())))
        self.train_p_id_list = sorted(list(set(self.p_a_dir_dict_train.keys())))
        self.test_p_id_list = sorted(list(set(self.p_a_dir_dict_test.keys())))

        self.author_num = len(self.a_p_dir_dict_train.keys())
        assert len(self.a_p_dir_dict_train) == len(self.author_train_map)

        # map authors to their idxs
        self.p_a_idx_dir_dict_train = {}
        for paper, authors in self.p_a_dir_dict_train.items():
            for author in authors:
                author_idx = self.author_train_map[author]
                self.p_a_idx_dir_dict_train.setdefault(paper, []).append(author_idx)

        self.p_a_idx_dir_dict_test = {}
        for paper, authors in self.p_a_dir_dict_test.items():
            for author in authors:
                author_idx = self.author_train_map[author]
                self.p_a_idx_dir_dict_test.setdefault(paper, []).append(author_idx)



        # indirect paper-author relation from heterogeneous walk
        def p_a_indir_set():
            p_a_indir_dict_train = dict()
            indir_relation_f = ["/metapathwalk_apa" + self.file_name + ".txt",
                                "/metapathwalk_apvpa" + self.file_name + ".txt",
                                "/metapathwalk_appa" + self.file_name + ".txt"]
            for f_index in range(len(indir_relation_f)):
                f_name = indir_relation_f[f_index]
                neigh_f = open(self.data_path + f_name, "r")
                print("Reading {}".format(self.data_path + f_name))
                for line in neigh_f:
                    line = line.strip()
                    path = re.split(' ', line)
                    for k in range(len(path)):
                        curr_node = path[k]
                        if curr_node[0] == 'p':
                            for w in range(k - self.args.window, k + self.args.window + 1):
                                if w >= 0 and w < len(path) and w != k:
                                    neigh_node = path[w]
                                    node = int(path[w][1:])
                                    if neigh_node[0] == 'a' and node not in self.p_a_dir_dict_train[int(curr_node[1:])]:
                                        p_a_indir_dict_train.setdefault(int(curr_node[1:]), []).append(node)
                neigh_f.close()
            return p_a_indir_dict_train

        if self.args.embedder == 'camel':
            print("Creating indirect triples")
            self.p_a_indir_dict_train = p_a_indir_set()
            self.indir_len = sum(len(self.p_a_indir_dict_train[x]) for x in self.p_a_indir_dict_train)

            self.p_a_idx_indir_dict_train = dict()
            for paper, authors in self.p_a_indir_dict_train.items():
                for author in authors:
                    self.p_a_idx_indir_dict_train.setdefault(paper, []).append(self.author_train_map[author])
            print("Done creating indirect triples")

        def pair_context_with_author_set():
            pairs = dict()
            contexts = dict()
            labels = dict()
            metapaths_f = ["/metapathwalk_apa" + self.file_name + ".txt"]
            for f_name in metapaths_f:
                neigh_f = open(self.data_path + f_name, "r")
                print("Reading {}".format(self.data_path + f_name))
                lines = neigh_f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    path = re.split(' ', line)
                    for i in range(len(path)):
                        start_i = max(0, i - self.args.ctx_window)
                        end_i = min(len(path), i + self.args.ctx_window + 1)
                        for k in range(start_i, end_i - 1):
                            for j in range(k + 1, end_i):
                                pair = (path[k], path[j])
                                if (pair[0][0] == 'a' and pair[1][0] == 'p'):
                                    context = path[max(0, k):min(len(path),j + 1)]
                                    if len(context) <= 2:
                                        continue
                                    # represent an author by the papers that he wrote
                                    # (author, paper)
                                    author = int(pair[0][1:])
                                    author_as_papers = self.a_p_dir_dict_train[author]
                                    paper = int(pair[1][1:])

                                    pairs.setdefault(len(context),[]).append([[author_as_papers, self.author_train_map[author]], [paper]])
                                    contexts.setdefault(len(context),[]).append([[int(elem[1:])] if elem[0] == 'p' else [self.a_p_dir_dict_train[int(elem[1:])], self.author_train_map[int(elem[1:])]] for elem in context])
                                    if paper in author_as_papers:
                                        labels.setdefault(len(context), []).append(1)
                                    else:
                                        labels.setdefault(len(context), []).append(0)

                neigh_f.close()

            pair_context_label = (pairs, contexts, labels)

            return pair_context_label


        if self.args.embedder == 'TapEM':
            print("Creating pair context")
            self.pair_context_label = pair_context_with_author_set()
            print("Done Creating pair context")

        def load_p_content(path, word_n=100000):
            f = open(path, 'rb')
            p_content_set = pickle.load(f)
            f.close()
            def remove_unk(x):
                return [[1 if w >= word_n else w for w in sen] for sen in x]

            p_content, p_content_id = p_content_set
            p_content = remove_unk(p_content)
            p_content_set = (p_content, p_content_id)

            return p_content_set

        def load_word_embed(path):
            lines = open(path,"r").readlines()
            word_n = len(lines)
            # word_n = 54559
            word_dim = len(lines[0].split(" ")[1:])
            # word_dim = 300
            word_embed = np.zeros((word_n + 2, word_dim))
            for line in lines:
                index = int(line.split()[0])
                embed = np.array(line.split()[1:])
                word_embed[index] = embed

            return word_embed

        # text content (e.g., abstract) of paper and pretrain word embedding
        print("Reading {}".format(self.data_path + '/content'+self.file_name+'.pkl'))
        self.p_content, self.p_content_id = load_p_content(path=self.data_path + '/content'+self.file_name+'.pkl')
        print("Reading {}".format(self.data_path + '/word_embedding'+self.file_name+'.txt'))
        self.word_embed = load_word_embed(path=self.data_path + '/word_embedding'+self.file_name+'.txt')

        self.print_stats()
        # print("Generate neg ids")
        # self.gen_evaluate_neg_ids()


    def p_a_a_dir_next_batch(self):
        p_a_a_dir_list_batch = []
        for p_id, a_ids in self.p_a_idx_dir_dict_train.items():
            for a_pos in a_ids:
                a_neg = random.randint(0, self.author_num - 1)
                while (a_neg in a_ids):
                    a_neg = random.randint(0, self.author_num - 1)
                # triple = [p_id, int(a_pos[1:]), a_neg]
                triple = [p_id, a_pos, a_neg]
                p_a_a_dir_list_batch.append(triple)
        return torch.LongTensor(p_a_a_dir_list_batch)


    def p_a_a_dir_next_batch_negsample(self):
        p_a_a_dir_list_batch = []
        for p_id, a_ids in self.p_a_idx_dir_dict_train.items():
            for a_pos in a_ids:
                for _ in range(self.args.num_dir_neg):
                    a_neg = random.randint(0, self.author_num - 1)
                    while (a_neg in a_ids):
                        a_neg = random.randint(0, self.author_num - 1)
                    # triple = [p_id, int(a_pos[1:]), a_neg]
                    triple = [p_id, a_pos, a_neg]
                    p_a_a_dir_list_batch.append(triple)

        shuffle(p_a_a_dir_list_batch)
        return torch.LongTensor(p_a_a_dir_list_batch)

    def p_a_a_indir_next_batch(self):
        p_a_a_indir_list_batch = []
        p_threshold = float(self.dir_len) / self.indir_len + 3e-3
        for p_id, a_ids in self.p_a_idx_indir_dict_train.items():
            for a_pos in a_ids:
                if random.random() < p_threshold:
                    a_neg = random.randint(0, self.author_num - 1)
                    while (a_neg in a_ids):
                        a_neg = random.randint(0, self.author_num - 1)
                    triple = [p_id, a_pos, a_neg]
                    p_a_a_indir_list_batch.append(triple)
        return torch.LongTensor(p_a_a_indir_list_batch)


    # this is for dir
    def gen_content_batch(self, triple_batch):
        p_c_data = []
        for i in range(len(triple_batch)):
            c_temp = self.p_content[triple_batch[i][0]]
            p_c_data.append(c_temp)
        return p_c_data

    # this is a general function to get contents from a list of papers
    def gen_content(self, papers):
        p_c_data = []
        for paper in papers:
            c_temp = self.p_content[paper]
            p_c_data.append(c_temp)
        return p_c_data

    # used when we need negative ids for evaluation
    def gen_evaluate_neg_ids(self):
        print("Writing to {}".format(self.data_path + "/paper_author_neg_ids"+self.file_name+".txt"))
        p_a_neg_ids_f = open(self.data_path + "/paper_author_neg_ids"+self.file_name+".txt", "w")
        for paper, a_idxs in self.p_a_idx_dir_dict_test.items():
            p_a_neg_ids_f.write(str(paper) + ":")
            neg_num = 100 - len(a_idxs)
            for _ in range(neg_num):
                neg_id = random.randint(0, self.author_num - 1)
                # while (neg_id in self.a_idx_p_dir_dict_test):
                while (neg_id in a_idxs) or (neg_id not in self.a_idx_p_dir_dict_train):
                    neg_id = random.randint(0, self.author_num - 1)
                p_a_neg_ids_f.write(str(neg_id) + ",")
            p_a_neg_ids_f.write("\n")
        p_a_neg_ids_f.close()


    def print_stats(self):
        print("Data Statistics")
        numPapers = len(self.p_a_dir_dict_train) + len(self.p_a_dir_dict_test)
        numAuthors = len(self.author_train_map)

        reference_file = self.data_path + "/paper_paper{}.txt".format(self.file_name)
        f = open(reference_file)
        lines = f.readlines()
        numCitations = len(lines)

        conf_file = self.data_path + "/paper_conf{}.txt".format(self.file_name)
        f = open(conf_file)
        lines = f.readlines()
        self.p_conf_dict = {}
        for line in lines:
            (paper, conf) = line.strip().split("\t")
            self.p_conf_dict.setdefault(int(paper), -1)
            self.p_conf_dict[int(paper)] = int(conf)

        numVenues = len(set(self.p_conf_dict.values()))

        avePaperPerAuthor = (sum([len(elems) for elems in self.p_a_dir_dict_train.values()]) + sum([len(elems) for elems in self.p_a_dir_dict_test.values()])) / numPapers

        print("=================================================================================================")
        print("numAuthors: {} | numPapers: {} | numVenues: {} | numCitations: {} | Authors/Papers: {}"
              .format(numAuthors, numPapers, numVenues, numCitations, avePaperPerAuthor))
        print("=================================================================================================")











