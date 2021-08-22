import torch

from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertRE
from dataset import Dataset
from preprocess import generate_examples

import config

def eval(model, test_dataset, n_test_relation):
    golds = []
    predicteds = []
    examples = []
    for batch in test_dataset.get_tqdm(device, False):
        data_x, data_y, data_e1, data_e2, exp = batch

        golds.extend(data_y.cpu().numpy())

        logits, loss = model(data_x, data_e1, data_e2, data_y)
        predicted_y = torch.argmax(logits, -1)
        predicteds.extend(list(predicted_y.cpu().numpy()))

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

        # if g > 0:  ### incorrect ....
        #     n_gold += 1

        if p > 0 and g > 0:  ### note here ..... 
            n_predict += 1
        if p == g and g > 0:
            n_correct += 1
    
    precision = n_correct/ (n_predict + 1e-200)
    recall = n_correct/n_gold
    f1 =  2* precision * recall / (precision + recall + 1e-200)
    print('###################', n_gold, n_predict, n_correct, precision, recall, f1)


if __name__ == '__main__':

    device = 'cuda'

    model = BertRE(config.bert_dir, 10)
    model.to(device)

    lr = 3e-5
    num_total_steps = 50
    num_warmup_steps = 0
    batch_size = 1

    train_examples, dev_examples, test_examples, n_train_relation, n_dev_relation, n_test_relation = generate_examples()

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps * int(len(train_examples) / batch_size))

    train_dataset = Dataset(train_examples)
    dev_dataset = Dataset(dev_examples)
    test_dataset = Dataset(test_examples)

    for i in range(0, num_total_steps):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, data_y, data_e1, data_e2, _ = batch        
            logits, loss = model(data_x, data_e1, data_e2, data_y)
                
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        model.eval()
        eval(model, dev_dataset, n_test_relation)
