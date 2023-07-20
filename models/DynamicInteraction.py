import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

from models.Cells import RectifiedIdentityCell, IntraModelReasoningCell, GlobalLocalGuidanceCell, CrossModalRefinementCell


from models.Cells import RectifiedActivationUnit, FeatureSemanticReasoningUnit, CrossmodalFusionUnit

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedActivationUnit(args, num_out_path)
        self.imrc = FeatureSemanticReasoningUnit(args, num_out_path)
        self.cmrc = CrossmodalFusionUnit(args, num_out_path)

    def forward(self, rgn, img, wrd, stc, stc_lens):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        emb_lst[0], path_prob[0] = self.ric(wrd)#torch.Size([16, 80, 1024]) torch.Size([16, 3])
        emb_lst[1], path_prob[1] = self.imrc(wrd, stc_lens=stc_lens)#torch.Size([16, 80, 1024]) torch.Size([16, 3])
        emb_lst[2], path_prob[2] = self.cmrc(rgn, img, wrd, stc, stc_lens)#torch.Size([16, 80, 1024]) torch.Size([16, 3])

        gate_mask = (sum(path_prob) < self.threshold).float() 
        all_path_prob = torch.stack(path_prob, dim=2)  
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0] #torch.Size([16, 80, 1024])
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])

                cur_emb = emb_lst[j]

                res = res + cur_path * cur_emb
            res = res + skip_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

class DynamicInteraction_Layer(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedActivationUnit(args, num_out_path)
        self.imrc = FeatureSemanticReasoningUnit(args, num_out_path)
        self.cmrc = CrossmodalFusionUnit(args, num_out_path)
        

    def forward(self, ref_wrd, rgn, img, wrd, stc, stc_lens):
        assert len(ref_wrd) == self.num_cell and ref_wrd[0].dim() == 3

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        emb_lst[1], path_prob[1] = self.imrc(ref_wrd[1], stc_lens)
        emb_lst[2], path_prob[2] = self.cmrc(rgn, img, ref_wrd[2], stc, stc_lens)
        
        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)  
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()   
            all_path_prob = torch.stack(path_prob, dim=2)   
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:,  i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:,  i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

    


