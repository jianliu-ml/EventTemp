import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.distance import CosineSimilarity
from transformers import BertModel

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

from model_rgcn import RGCN


class BertRE(nn.Module):
    def __init__(self, pre_trained, y_num):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(pre_trained)
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, y_num)
        # self.span_extractor = EndpointSpanExtractor(combination='x', input_dim=self.bert.config.hidden_size)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size)


    def forward(self, data_x, data_e1, data_e2, data_y):

        outputs = self.bert(data_x)
        bert_enc = outputs[0]

        enc1 = self.span_extractor(bert_enc, data_e1.unsqueeze(0))
        enc2 = self.span_extractor(bert_enc, data_e2.unsqueeze(0))

        temp = torch.cat([enc1, enc2], dim=-1)
        temp = temp.squeeze(0)

        logits = self.fc(temp)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits.view(-1, self.y_num), data_y.view(-1))

        return logits, loss



class BertREGraph(nn.Module):
    def __init__(self, pre_trained, y_num):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(pre_trained)
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, y_num)
        # self.span_extractor = EndpointSpanExtractor(combination='x', input_dim=self.bert.config.hidden_size)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size)

        self.rgcn = RGCN(
            768,  # input
            10,   # mid
            768,  # output
            8,    # n_edge
            -1,
            5,    # layers
            0.5,
            True
        )


    def forward(self, data_x, data_e1, data_e2, data_y, graphs, edge_types):

        outputs = self.bert(data_x)
        bert_enc = outputs[0]

        rgcn_temp = []
        for x, y, z in zip(graphs, bert_enc, edge_types):
            rgcn_temp.append(self.rgcn(x, y, z, None))
        rgcn_temp = torch.stack(rgcn_temp, dim=0)

        # print(bert_enc.size())
        # print(rgcn_temp.size())

        bert_enc = bert_enc + 0.2 * rgcn_temp

        enc1 = self.span_extractor(bert_enc, data_e1.unsqueeze(0))
        enc2 = self.span_extractor(bert_enc, data_e2.unsqueeze(0))

        temp = torch.cat([enc1, enc2], dim=-1)
        temp = temp.squeeze(0)

        # print(temp.size())

        logits = self.fc(temp)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits.view(-1, self.y_num), data_y.view(-1))

        return logits, loss