import torch
import dgl

from tqdm import tqdm, trange
from scipy.stats import entropy
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertREGraph
from dataset import DatasetGraphImputation, DatasetGraphTest
from preprocess import generate_examples

import config
import numpy as np

torch.autograd.set_detect_anomaly(True)

from util import save_model, load_model

def list_to_entropy(l): # less is better
    a = {}
    for e in l:
        a.setdefault(e, 0)
        a[e] += 1
    prob = [a[x]/len(l) for x in a]
    return entropy(prob)

def compute_uncertainty(model, data_x, data_e1, data_e2, data_y, graphs, edge_types):
    model_copy = model
    res = []
    for i in range(5):
        softmax_logits, _ = model_copy(data_x, data_e1, data_e2, data_y, graphs, edge_types) # batchsize class
        result = torch.argmax(softmax_logits, -1).cpu().numpy()
        res.append(result)
    res = np.asarray(res)  # 5, batch_size
    res = np.transpose(res, (1, 0))  # batchsize, 5
    
    result = [list_to_entropy(x) for x in res]  ### -?????
    return np.asarray(result)


def eval(model, test_dataset, device, n_test_relation):
    with torch.no_grad():
        golds = []
        predicteds = []
        examples = []

        for batch in test_dataset.get_tqdm(device, False):
            data_x, data_y, data_e1, data_e2, exp, src1, tgt1, edge1 = batch
            golds.extend(data_y.cpu().numpy())

            src = list(range(len(data_x[0]))) + src1
            tgt = list(range(len(data_x[0]))) + tgt1
            edge_type = [0] * len(data_x[0]) + edge1

            alread_predicted = {}
            temp_predicted = [0] * len(data_y)
            
            for _ in trange(len(data_y)):  ###
                
                u, v = np.asarray(src), np.asarray(tgt)
                g = dgl.DGLGraph((u, v))
                graphs = [g]
                
                f = torch.LongTensor
                edge_types = [f(edge_type).to(device)]

                model.eval()
                logits, loss = model(data_x, data_e1, data_e2, data_y, graphs, edge_types)
                predicted_y = torch.argmax(logits, -1)

                model.train()
                uncty = compute_uncertainty(model, data_x, data_e1, data_e2, data_y, graphs, edge_types)
                arg_sort = np.argsort(uncty)

                for p in arg_sort:  ## position
                    if not p in alread_predicted:
                        alread_predicted[p] = 1

                        e1, e2 = data_e1[p], data_e2[p]
                        p_label = predicted_y[p]

                        src.append(e1[0])
                        tgt.append(e2[0])
                        edge_type.append(p_label)

                        temp_predicted[p] = p_label

                        break

            predicteds.extend(temp_predicted)
            examples.extend(exp)

        event_ids = []
        for elem in examples:
            exp_id, tokens, events, subword_ids, ann_event_relations, exp, exp2, _ = elem
            for e in exp2:
                event_ids.append((exp_id, e[0], e[1]))
        
        p_set = {}

        n_gold = n_test_relation
        n_predict, n_correct = 0, 0

        for e, g, p in zip(event_ids, golds, predicteds):
            if e in p_set:
                continue
            p_set[e] = 1

            # p = 3

            # if g > 0:
            #     n_gold += 1
            if p > 0 and g > 0:  ### note here ..... 
                n_predict += 1
            if p == g and g > 0:
                n_correct += 1
        
        precision = n_correct/ (n_predict + 1e-200)
        recall = n_correct/n_gold
        f1 =  2* precision * recall / (precision + recall + 1e-200)
 
        print('###################', n_gold, n_predict, n_correct, precision, recall, f1)

        return n_gold, n_predict, n_correct, precision, recall, f1


if __name__ == '__main__':

    device = 'cuda'

    model = BertREGraph(config.bert_dir, 10)
    model.to(device)

    lr = 3e-5
    num_total_steps = 5000
    num_warmup_steps = 0
    batch_size = 1

    train_examples, dev_examples, test_examples, n_train_relation, n_dev_relation, n_test_relation = generate_examples()

    print(train_examples[0])

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps * int(len(train_examples) / batch_size))

    train_dataset = DatasetGraphImputation(train_examples)  # for training
    dev_dataset = DatasetGraphTest(dev_examples)
    test_dataset = DatasetGraphTest(test_examples) # for testing

    max_f1 = 0

    for i in range(0, num_total_steps):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, data_y, data_e1, data_e2, graphs, edge_types, example = batch        
            logits, loss = model(data_x, data_e1, data_e2, data_y, graphs, edge_types)
                
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if i % 5 == 0:
            model.eval()
            n_gold, n_predict, n_correct, precision, recall, f1 = eval(model, dev_dataset, device, n_test_relation)

            if f1 > max_f1:
                max_f1 = f1
                print('saving....', f1)
                save_model(model, 'models/best_model')
