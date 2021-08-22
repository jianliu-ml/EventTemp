import copy
import dgl
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm

import config


class Dataset(object):
    def __init__(self, dataset, batch_size=1):
        super().__init__()

        self.batch_size = batch_size
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        ## exp_id, tokens, events, subword_ids, ann_event_relations, exp

        data_x = list()
        data_y = list()
        e1, e2 = list(), list()

        example = []

        for data in batch:
            data_x.append(data[3])

            for elem in data[5]:
                e1.append(elem[0])
                e2.append(elem[1])
                data_y.append(elem[2])

            example.append(data)

        f = torch.LongTensor
        
        data_x = f(data_x)
        data_y = f(data_y)
        data_e1 = f(e1)
        data_e2 = f(e2)

        return [data_x.to(device),  
                data_y.to(device),
                data_e1.to(device),
                data_e2.to(device),
                example]






class DatasetGraphImputation(object):
    def __init__(self, dataset, batch_size=1):
        super().__init__()

        self.batch_size = batch_size
        self.construct_index(dataset)
        self.imputation_ratio = 0.2

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        ## exp_id, tokens, events, subword_ids, ann_event_relations, exp, exp2

        data_x = list()
        data_y = list()
        e1, e2 = list(), list()

        graphs = list()
        edge_types = list()

        example = []

        for data in batch:
            data_x.append(data[3])

            n_relation = len(data[5])
            all_relations = list(range(n_relation))
            choice = [random.randint(0, n_relation-1)]
            choice = choice + random.choices(all_relations, k=int(self.imputation_ratio * n_relation))
            ## choice contains elements to prediction

            src = []
            tgt = []
            edge_type = []

            for idx, elem in enumerate(data[5]):
                if idx in choice:
                    e1.append(elem[0])
                    e2.append(elem[1])
                    data_y.append(elem[2])
                else:
                    src.append(elem[0][0])  ## only head
                    tgt.append(elem[1][0])
                    edge_type.append(elem[2])
            
            src1 = []
            tgt1 = []
            edge1 = []

            for chain in data[7]:
                for elem1 in chain:
                    for elem2 in chain:
                        if elem1 != elem2:
                            src1.append(elem1[0])
                            tgt1.append(elem2[0])
                            edge1.append(6)        ### 6 stands for entity coref...

            src = src + list(range(len(data[3]))) + src1
            tgt = tgt + list(range(len(data[3]))) + tgt1
            u, v = np.asarray(src), np.asarray(tgt)
            g = dgl.DGLGraph((u, v))
            graphs.append(g)

            edge_type = edge_type + [0] * len(data[3]) + edge1
            edge_types.append(edge_type)

            example.append(data)

        f = torch.LongTensor
        
        data_x = f(data_x)
        data_y = f(data_y)
        data_e1 = f(e1)
        data_e2 = f(e2)

        return [data_x.to(device),  
                data_y.to(device),
                data_e1.to(device),
                data_e2.to(device),
                graphs,
                f(edge_types).to(device),
                example]




class DatasetGraphTest(object):
    def __init__(self, dataset, batch_size=1):
        super().__init__()

        self.batch_size = batch_size
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        data_x = list()
        data_y = list()
        e1, e2 = list(), list()

        example = []

        src1 = []
        tgt1 = []
        edge1 = []

        for data in batch:
            data_x.append(data[3])

            for elem in data[5]:
                e1.append(elem[0])
                e2.append(elem[1])
                data_y.append(elem[2])

            example.append(data)



            for chain in data[7]:
                for elem1 in chain:
                    for elem2 in chain:
                        if elem1 != elem2:
                            src1.append(elem1[0])
                            tgt1.append(elem2[0])
                            edge1.append(6)        ### 6 stands for entity coref...

        f = torch.LongTensor
        
        data_x = f(data_x)
        data_y = f(data_y)
        data_e1 = f(e1)
        data_e2 = f(e2)

        return [data_x.to(device),  
                data_y.to(device),
                data_e1.to(device),
                data_e2.to(device),
                example,
                src1,
                tgt1,
                edge1]


# src = [x[0] for x in data[5]] + list(range(512))
# dst = [x[1] for x in data[5]] + list(range(512))

# u, v = np.asarray(src), np.asarray(dst)
# g = dgl.DGLGraph((u, v))
# graphs.append(g)

# edge_type = [x[2] + 1 for x in data[5]] + [0] * 512
# edge_types.append(edge_type)



if __name__ == "__main__":
    
    from model import BertREGraph
    from preprocess import generate_examples

    train_examples, dev_examples, test_examples = generate_examples()

    print(len(train_examples), len(dev_examples), len(test_examples))

    model = BertREGraph(config.bert_dir, 20)
    train_dataset = DatasetGraphImputation(train_examples)
    for batch in train_dataset.reader('cpu', False):
        data_x, data_y, data_e1, data_e2, graphs, edge_types, example = batch        
        # print(data_y)
        # print(data_e1)
        # print(data_e2)
        model(data_x, data_e1, data_e2, data_y, graphs, edge_types)
        print('Here')