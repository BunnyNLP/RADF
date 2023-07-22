import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
from torchvision import models


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math

from models.InteractionModule import InteractionModule
from .loss import ContrastiveLoss
from models.TextNet import EncoderText
from models.VisNet import EncoderImage


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
from .DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer 

from .attention import MultiHeadAttention


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

class ImageModel(nn.Module):
    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

        freeze_layers(self.resnet)

        self.fc = nn.Linear(2048, args.embed_size)
        self.dropout = nn.Dropout(p=0.5)# dropout训练
    
    def forward(self, x): #torch.Size([16, 3, 224, 224]), torch.Size([16, 3, 3, 224, 224])

        for name, layer in self.resnet.named_children():
            x = layer(x)    # (bsz, 256, 56, 56)
            if name == 'avgpool':
                img = self.fc(x.squeeze(-1,-2))
                # img = self.dropout(img)
                # img = F.normalize(img, p=2, dim=-1)
                return img

class PrefixEncoder(torch.nn.Module):
    def __init__(self,seq_len,dim_ebd,num_layer):
        '''
        seq_len : The length of prompt
        dim_ebd : the dimension of the bert model
        num_layer : the number of hidden layer in BERT
        '''
        super().__init__()
        # self.embedding = torch.nn.Embedding(seq_len, dim_ebd)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(dim_ebd, dim_ebd),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_ebd, num_layer * 2 * dim_ebd)
        ).cuda()
    def forward(self, prefix):#bsz,80,1024
        # prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix)
        return past_key_values
    


class RADFREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args, num_layer_routing=3, num_cells=3, path_hid=128):
        super(RADFREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        freeze_layers(self.bert)#冻结参数

        self.image_model = ImageModel(args)

        ############################## Dynamic Router Patameters ############################## 
        self.txt_enc = EncoderText(args,self.bert)
        self.img_enc = EncoderImage(args.img_dim, args.embed_size, args.finetune, use_abs=args.use_abs, no_imgnorm=args.no_imgnorm, drop=args.drop)
        
        self.num_cells = num_cells
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(self.args, num_cells, num_cells)
        self.dynamic_itr_l1 = DynamicInteraction_Layer(self.args, num_cells, num_cells)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(self.args, num_cells, 1)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(self.args.embed_size)

        ######################################## Classifier ######################################## 
        self.lstm = torch.nn.LSTM(1024,512,batch_first=True,bidirectional=True,dropout=0.1)
        
        self.fc_entity = nn.Linear(768,1024)
        self.dropout_ent = nn.Dropout(p=0.5)
        self.mha = MultiHeadAttention(n_head=16, query_input_dim=1024, key_input_dim=1024 , value_input_dim=1024, query_hidden=1024, key_hidden=1024, value_hidden=1024, output_dim=1024)#n_head, query_input_dim, key_input_dim, value_input_dim, query_hidden, key_hidden, value_hidden, output_dim

        self.fc_prompt = nn.Linear(80,10)
        self.fc_prompt_ = nn.Linear(1024,768)
        self.prompt_encoder = PrefixEncoder(10, self.bert.config.hidden_size ,self.bert.config.num_hidden_layers)

        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)

    def get_prompt(self, batch_size, x):
        '''x : bsz, 10, 768'''
        past_key_values = self.prompt_encoder(x)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.bert.config.num_hidden_layers * 2,
            self.bert.config.num_attention_heads,
            self.bert.config.hidden_size//self.bert.config.num_attention_heads
                )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values, seqlen
    


    def forward(
        self,
        input_ids=None, #torch.Size([16, 80])
        attention_mask=None, #torch.Size([16, 80])
        token_type_ids=None, #torch.Size([16, 80])
        labels=None, #torch.Size([16])
        images=None, #torch.Size([16, 3, 224, 224]) 只是经过预处理的图像
        # aux_imgs=None, #torch.Size([16, 3, 3, 224, 224])
        img_feat = None, #torch.Size([4,1024])
        head_enc = None,
        tail_enc = None        
    ):
        '''
        rgn : torch.Size([32, 36, 256]) -> torch.Size([bsz, 4 , 1024]) -> torch.Size([16, 4 , 1024])
        img : torch.Size([32, 256]) -> torch.Size([16, 1024])
        wrd : torch.Size([64, 32, 256]) -> torch.Size([bsz, max_len , dim]) -> torch.Size([16, 80 ,1024])
        stc : torch.Size([64, 256] -> torch.Size([16, 1024]))
        stc_lens : 64 * 1 -> 16 * 1
        '''


        _ , rgn = self.img_enc(img_feat) #object-level feature -> torch.Size([16, 4, 1024]) 下划线是通过平均池化求出来的全局图片向量
        img = self.image_model(images)#torch.Size([16, 1024])
        stc, wrd = self.txt_enc(input_ids, attention_mask, token_type_ids)
        stc_lens = input_ids.shape[0]


        #Head and Tail Entity
        head_out = self.bert(input_ids = torch.stack(head_enc['input_ids'],dim=1).cuda(), token_type_ids=torch.stack(head_enc['token_type_ids'],dim=1).cuda(), attention_mask=torch.stack(head_enc['attention_mask'],dim=1).cuda())
        tail_out = self.bert(input_ids = torch.stack(tail_enc['input_ids'],dim=1).cuda(), token_type_ids=torch.stack(tail_enc['token_type_ids'],dim=1).cuda(), attention_mask=torch.stack(tail_enc['attention_mask'],dim=1).cuda())
        head_o = head_out.pooler_output#torch.Size([16, 768])
        tail_o = tail_out.pooler_output#torch.Size([16, 768])
        entity_emb = torch.stack([head_o,tail_o],dim=1)


        #DIME
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(rgn, img, wrd, stc, stc_lens)
        pairs_emb_lst, paths_l1 = self.dynamic_itr_l1(pairs_emb_lst, rgn, img, wrd, stc, stc_lens)
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, rgn, img, wrd, stc, stc_lens)


        #Prompt tuning
        bsz = input_ids.size(0)
        prompt_guids = pairs_emb_lst[0].permute(0,2,1)#bsz,1024,80
        prompt_guids = self.fc_prompt(prompt_guids)#bsz,1024,10
        prompt_guids = prompt_guids.permute(0,2,1)#bsz,10,1024
        prompt_guids = self.fc_prompt_(prompt_guids)#bsz,10,768
        prompt_guids, seq_len = self.get_prompt(bsz, prompt_guids)


        # seq_len = prompt_guids.size(1)
        prompt_guids_mask = torch.ones(bsz,seq_len).cuda()

        if self.args.use_prompt:
            prompt_guids = prompt_guids
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).cuda()
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        
        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # import pdb;pdb.set_trace()
        logits = self.classifier(entity_hidden_state)
        
        #普通分类头
        # lstm_output, (hn, cn) = self.lstm(pairs_emb_lst[0])#torch.Size([bsz, 80, 1024])
        # entity_joint_emb = self.fc_entity(entity_emb)#torch.Size(bzs, 2, 1024])
        # entity_joint_emb = self.dropout_ent(entity_joint_emb)
        # att, att_out = self.mha(entity_joint_emb, pairs_emb_lst[0], pairs_emb_lst[0])
        # bsz = att_out.shape[0]
        # logits = att_out.view(bsz,-1) 
        # logits = torch.cat([logits, stc ],dim=-1)     
        # logits = self.classifier(logits)
        

        # n_img, n_stc = paths_l2.size()[:2]#torch.Size([16, 1, 3]) -> torch.Size([16, 1])

        # paths_l0 = paths_l0.contiguous().view(n_img, -1).unsqueeze(1).expand(-1, n_stc, -1)
        # paths_l1 = paths_l1.view(n_img, n_stc, -1)
        # paths_l2 = paths_l2.view(n_img, n_stc, -1)
        # paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)
        # paths = paths.mean(dim=1) # (n_img, total_paths)

            
        # paths = self.path_mapping(paths)
        # paths = F.normalize(paths, dim=-1)
        # sim_paths = paths.matmul(paths.t())

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits
    




class RADFNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(RADFNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

        self.num_labels  = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)    # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result
