import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import OrderedDict 

# import tokenization
from .BERT import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertConfig, BertTokenizer, BertModel
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class EncoderText(nn.Module):
    """
    """
    def __init__(self, opt, bert):
        super(EncoderText, self).__init__()
        self.opt = opt

        #使用huggingface的包不需要转换
        self.bert =  bert
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        embed_size = opt.embed_size


        # 1D-CNN
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = opt.embed_size
        bert_hid = bert_config.hidden_size
        self.fc = nn.Linear(bert_hid, opt.embed_size)
        self.fc_stc = nn.Linear(bert_hid, opt.embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, bert_hid)) for K in Ks])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.mapping = nn.Linear(len(Ks)*out_channel, opt.embed_size)

    def forward(self, inputids, attention_mask, token_type_ids ):
        
        results = self.bert(input_ids = inputids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_encoder_layers = results.hidden_states#13个元素的元组，表示每一层的输出，需output_hidden_states=True -> 12 *torch.Size([16, 80, 768])
        pooled_output = results.pooler_output#序列第一个CLStouken的输出 -> torch.Size([16, 768])

        # x = all_encoder_layers[-1].unsqueeze(1)#取最后一个隐藏层
        x = results.last_hidden_state#torch.Size([16, 80, 768])
        wrd_emb = self.fc(x)
        wrd_emb = F.normalize(wrd_emb, p=2, dim=-1)

        cls_out = results.pooler_output#torch.Size([16, 768])
        stc_emb = self.fc_stc(cls_out)
        stc_emb = F.normalize(stc_emb, p=2, dim=-1)


        # if self.training:
        #     bert_emb = all_encoder_layers[-1].detach().mean(dim=1)
        #     bert_emb = F.normalize(bert_emb, dim=-1)# torch.Size([16, 768])
        # x_emb = self.fc(all_encoder_layers[-1])#torch.Size([16, 80, 1024])
        # x1 = F.relu(self.convs1[0](x)).squeeze(3)#torch.Size([16, 1024, 80])
        # x2 = F.relu(self.convs1[1](F.pad(x, (0, 0, 0, 1)))).squeeze(3)#torch.Size([16, 1024, 80])
        # x3 = F.relu(self.convs1[2](F.pad(x, (0, 0, 1, 1)))).squeeze(3)#torch.Size([16, 1024, 80])
        # x = torch.cat([x1, x2, x3], dim=1)#torch.Size([16, 3072, 80])
        # x = x.transpose(1, 2)#torch.Size([16, 80, 3072])
        # word_emb = self.mapping(x)
        # word_emb = word_emb + x_emb#torch.Size([16, 80, 1024])
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [x1, x2, x3]]
        # x = torch.cat(x, 1)

        # txt_emb = self.mapping(x)
        # txt_emb = txt_emb + x_emb.mean(1)
        # txt_emb = F.normalize(txt_emb, p=2, dim=-1)
        # word_emb = F.normalize(word_emb, p=2, dim=-1)
        # if self.training:
        #     return (txt_emb, bert_emb), word_emb
        # else:
        #     return txt_emb, word_emb
        return stc_emb, wrd_emb
    
    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in own_state.items():
            if name in state_dict:
                new_state[name] = state_dict[name]
            else:
                new_state[name] = param
        super(EncoderText, self).load_state_dict(new_state)

