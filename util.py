from bs4 import BeautifulSoup, NavigableString
import re
import spacy
import neuralcoref
import requests
import torch

from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
lemmatizer = WordNetLemmatizer() 
ps = PorterStemmer() 

nlp = spacy.load('en')
nlp.tokenizer = nlp.tokenizer.tokens_from_list
neuralcoref.add_to_pipe(nlp)

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def precess_sentence(sentence):
    soup = BeautifulSoup(sentence)
    soup = soup.find('p')
    
    tokens = list()
    tags = list()
    for elem in soup.contents:
        if isinstance(elem, NavigableString):
            tokens.extend(elem.strip().split())
        else:
            eid = elem.get('eid')
            tid = elem.get('tid')
            
            eid = eid if eid else tid

            s = len(tokens)
            tokens.extend(elem.text.split())
            e = len(tokens)
            tags.append([eid, [s, e-1]])
        
    return tokens, tags


def process_document(doc_name):
    filein = open(doc_name)
    for _ in range(0, 5):
        filein.readline()
    results = list()
    for line in filein:
        if line.startswith('</TEXT>'): break
        sentence = '<p>' + line.strip() + '</p>'
        tokens, tags = precess_sentence(sentence)
        results.append([tokens, tags])

    tokens = []
    events = []
    for elem in results:
        offset = len(tokens)
        ts, es = elem
        tokens.extend(ts)
        for e in es:
            e[1][0] += offset
            e[1][1] += offset
            events.append(e)
    
    return [tokens, events]


def read_annotation(fn='data/TDDiscourse/TDDAuto/TDDAutoTest.tsv'):
    result = []
    with open(fn) as filein:
        for line in filein:
            fileds = line.strip().split()
            result.append(fileds)
    return result


def entity_coreference(tokens):
    doc = nlp(tokens)
    results = []
    for elem in doc._.coref_clusters:
        temp = [[mention.start, mention.end] for mention in elem.mentions]
        results.append(temp)
    return results


def to_coreference(filename='data/TimeBank-dense/test/CNN19980213.2130.0155.tml'):
    tokens, events = process_document(filename)
    res = []
    for e in events:
        e_id, [s, e] = e
        if e_id[0] == 'e':
            res.append('ABC-' + e_id + '-' + '_'.join(tokens[s:e+1]))
    print('ABC')
    res = ['1', '2005-01-18'] + res
    print('\t'.join(res))


def get_causal(event):
    obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
    relations = ['CapableOf', 'IsA', 'HasProperty', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent', 'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor', 'CreatedBy', 'MadeOf', 'Desires']
    res = []
    for e in obj['edges']:
        if e['rel']['label'] in relations:
            res.append([e['rel']['label'], e, event])
    return res


if __name__ == '__main__':

    tokens, events = process_document('data/TimeBank-dense/test/CNN19980213.2130.0155.tml')
    evs = []
    for e in events:
        e_id, [s, e] = e
        if e_id[0] == 'e':
            evs.append(' '.join(tokens[s:e+1]))
    for e in evs:
        print(e, ps.stem(e))
        res = get_causal(ps.stem(e))
        for r in res:
            print(r[2], r[0], r[1]['end']['label'])
            # print(r[2], r[0], r[1]['sources'])
            
        # print(res)
        # break


    # annos = read_annotation('data/TDDiscourse/TDDMan/TDDManTest.tsv')

    # print(annos[0])
    # print(tokens)

    # import requests
    # obj = requests.get('http://api.conceptnet.io/c/en/tsunami').json()
    # for e in obj['edges']:
    #     print(e['rel']['label'])
    #     # print(e)


    # import spacy
    # import neuralcoref
    # nlp = spacy.load('en')
    # nlp.tokenizer = nlp.tokenizer.tokens_from_list
    # neuralcoref.add_to_pipe(nlp)

    # doc = nlp('My sister has a dog . She loves him'.split(' '))

    # print(doc._.coref_clusters)
    # print(doc._.coref_clusters[0].mentions)
    # doc._.coref_clusters[0].mentions[-1]
    # doc._.coref_clusters[0].mentions[-1]._.coref_cluster.main

    # token = doc[-1]
    # print(token._.in_coref)
    # print(token._.coref_clusters[0].mentions[0])
    # print(token._.coref_clusters[0].mentions[0].start)  ##### Note Here
    # print(token._.coref_clusters[0].mentions[0].end)    ##### Note Here

    # span = doc[-1:]
    # span._.is_coref
    # span._.coref_cluster.main
    # span._.coref_cluster.main._.coref_cluster