import glob
import copy
import random

from transformers import BertTokenizer

import config
from util import process_document, read_annotation, entity_coreference

tokenizer = BertTokenizer.from_pretrained(config.bert_dir, do_lower_case=False)

def sentence_to_bert(tokens):
    subword_ids = list()
    subidx_to_origin = list()
    origin_to_subidx = list()

    for idx, word in enumerate(tokens):
        origin_to_subidx.append(len(subword_ids))
        
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)
        subword_ids.extend(sub_tokens)

        subidx_to_origin.extend([idx] * len(sub_tokens))

    return subword_ids, subidx_to_origin, origin_to_subidx


def process_document_to_bert_example(filename):
    tokens, events = process_document(filename)
    tokens = ['[CLS]'] + tokens
    for e in events:
        e[1][0] += 1
        e[1][1] += 1
    subword_ids, subidx_to_origin, origin_to_subidx = sentence_to_bert(tokens)
    return tokens, events, subword_ids, subidx_to_origin, origin_to_subidx


def _build_example_one_file(filename, all_annotations):
    tokens, events, subword_ids, subidx_to_origin, origin_to_subidx = process_document_to_bert_example(filename)
    ann_key = filename.split('/')[-1][:-4]
    ann_event_relations = list(filter(lambda x: x[0]==ann_key, all_annotations))

    event_set = set([x[1] for x in ann_event_relations] + [x[2] for x in ann_event_relations])
    events = list(filter(lambda x: x[0] in event_set, events))

    for event in events:
        s, e = event[1]
        s = origin_to_subidx[s]
        e = origin_to_subidx[e+1] - 1
        event[1] = [s, e]

    event_relations = {}
    for elem in ann_event_relations:
        event_relations[(elem[1], elem[2])] = elem[3]

    return tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations


    # print(ann_event_relations)


def _slide_window(examples, window_size=100, max_len=512):
    ## dropout subidx_to_origin, origin_to_subidx,
    result = []
    for exp_id, elem in enumerate(examples):
        tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations, entity_coref = elem
        if len(tokens) < max_len:
            result.append([exp_id, tokens, events, subword_ids, event_relations, entity_coref])
        else:
            idx = 0
            while True:
                start = idx * window_size
                end = min(start + max_len, len(subword_ids))
                subword_ids_temp = subword_ids[start:end]
                events_temp = []
                for event in copy.deepcopy(events):
                    event[1][0] -= start
                    event[1][1] -= start
                    if event[1][0] in range(0, end-start) and event[1][1] in range(0, end-start):
                        events_temp.append(event)
                
                event_set = set([x[0] for x in events_temp])
                key_set = list(filter(lambda x: x[0] in event_set and x[1] in event_set, event_relations))
                event_relations_temp = {}
                for key in key_set:
                    event_relations_temp[key] = event_relations[key]

                cor_results = []
                for cor_chain in copy.deepcopy(entity_coref):
                    temp = []
                    for elem in cor_chain:
                        elem[0] -= start
                        elem[1] -= start
                        if elem[0] in range(0, end-start) and elem[0] in range(0, end-start):
                            temp.append(elem)
                    if len(temp) > 2:
                        cor_results.append(temp)
                
                result.append([exp_id, tokens, events_temp, subword_ids_temp, event_relations_temp, cor_results])

                if end == len(subword_ids):
                    break
                idx += 1
            
    return result


def _coref_chain_to_sub_word_idx(coref_chain, origin_to_subidx):
    result = []
    for elem in coref_chain:
        temp = []
        for e in elem:
            x = [origin_to_subidx[e[0]], origin_to_subidx[e[1]]]
            temp.append(x)
        result.append(temp)
    return result


def build_all_example():

    # ann_train_file = 'data/TDDiscourse/TDDMan/TDDManTrain.tsv'
    # ann_dev_file = 'data/TDDiscourse/TDDMan/TDDManDev.tsv'
    # ann_test_file = 'data/TDDiscourse/TDDMan/TDDManTest.tsv'

    ann_train_file = 'data/TDDiscourse/TDDAuto/TDDAutoTrain.tsv'
    ann_dev_file = 'data/TDDiscourse/TDDAuto/TDDAutoDev.tsv'
    ann_test_file = 'data/TDDiscourse/TDDAuto/TDDAutoTest.tsv'

    ann_train = read_annotation(ann_train_file)
    ann_dev = read_annotation(ann_dev_file)
    ann_test = read_annotation(ann_test_file)

    # print(len(ann_train))
    # print(len(ann_dev))
    # print(len(ann_test))

    train_files = glob.glob("data/TimeBank-dense/train/*")
    dev_files = glob.glob("data/TimeBank-dense/dev/*")
    test_files = glob.glob("data/TimeBank-dense/test/*")

    train_examples = []
    dev_examples = []
    test_examples = []

    for train_fn in train_files:
        tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations = _build_example_one_file(train_fn, ann_train)
        entity_coref = entity_coreference(tokens)
        entity_coref = _coref_chain_to_sub_word_idx(entity_coref, origin_to_subidx)
        if len(events) > 0:
            train_examples.append([tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations, entity_coref])

    for dev_fn in dev_files:
        tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations = _build_example_one_file(dev_fn, ann_dev)
        entity_coref = entity_coreference(tokens)
        entity_coref = _coref_chain_to_sub_word_idx(entity_coref, origin_to_subidx)
        if len(events) > 0:
            dev_examples.append([tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations, entity_coref])

    for test_fn in test_files:
        tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations = _build_example_one_file(test_fn, ann_test)
        entity_coref = entity_coreference(tokens)
        entity_coref = _coref_chain_to_sub_word_idx(entity_coref, origin_to_subidx)
        if len(events) > 0:
            test_examples.append([tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations, entity_coref])

    return train_examples, dev_examples, test_examples


def _add_negative_examples(examples, ratio=1.0):

    res = []
    for elem in examples:
        exp_id, tokens, events, subword_ids, ann_event_relations, cor_results = elem
        exp = []
        exp2 = []
        for i in range(len(events)):
            for j in range(i+1, len(events)):
                e1_id, e1_pos = events[i]
                e2_id, e2_pos = events[j]

                l = 'n'
                if (e1_id, e2_id) in ann_event_relations:
                    l = ann_event_relations[(e1_id, e2_id)]

                if l != 'n' or ratio > 0.99:
                    exp.append([e1_pos, e2_pos, config.label_set[l]])
                    exp2.append([e1_id, e2_id, config.label_set[l]])
                elif random.random() < ratio:
                    exp.append([e1_pos, e2_pos, config.label_set[l]])
                    exp2.append([e1_id, e2_id, config.label_set[l]])

        if len(exp) == 0:
            print(exp_id, 'no event relation found')
            # print(events)
            # print(ann_event_relations)
            continue
        res.append([exp_id, tokens, events, subword_ids, ann_event_relations, exp, exp2, cor_results])
    return res


def _to_batch(examples, batch_size):
    res = []
    for elem in examples:
        exp_id, tokens, events, subword_ids, ann_event_relations, exp = elem
        idx = 0
        while True:
            start = idx * batch_size
            end = min((idx +  1) * batch_size, len(exp))
            exp_temp = exp[start:end]
            res.append([exp_id, tokens, events, subword_ids, ann_event_relations, exp_temp])
            
            if end == len(exp):
                break
            idx += 1
    return res




def generate_examples():
    train_examples, dev_examples, test_examples = build_all_example()

    n_train_relation = sum([len(example[5]) for example in train_examples])
    n_dev_relation = sum([len(example[5]) for example in dev_examples])
    n_test_relation = sum([len(example[5]) for example in test_examples])



    # tokens, events, subword_ids, subidx_to_origin, origin_to_subidx, event_relations = train_examples[0]
    # print(len(subword_ids))
    # print(events)

    train_examples = _slide_window(train_examples)  # slide_window, add idx, remove subidx_to_origin, origin_to_subidx
    dev_examples = _slide_window(dev_examples)
    test_examples = _slide_window(test_examples)

    train_examples = _add_negative_examples(train_examples, 0.000005) # to do, negative sampling?
    dev_examples = _add_negative_examples(dev_examples)
    test_examples = _add_negative_examples(test_examples)

    return train_examples, dev_examples, test_examples, n_train_relation, n_dev_relation, n_test_relation



if __name__ == '__main__':
 
    train_examples, dev_examples, test_examples = generate_examples()

    # exp_id, tokens, events, subword_ids, ann_event_relations, exp = train_examples[4]
    # print(exp_id, len(subword_ids))
    # print(events)


    # for elem in train_examples + dev_examples + test_examples:
    #     exp_id, tokens, events, subword_ids, ann_event_relations, exp = elem
    #     # print(exp_id, events)
    #     # print(subword_ids)
    #     # print(ann_event_relations)
    #     print(len(events))
    #     print(len(ann_event_relations))
    #     print(len(exp))
    #     print(exp[0])
    #     print(exp[1])
    #     print('---')

    tokens, events, subword_ids, subidx_to_origin, origin_to_subidx = process_document_to_bert_example('data/TimeBank-dense/test/CNN19980213.2130.0155.tml')
    
    print(entity_coreference(tokens))
    print(tokens)
    print(events)

