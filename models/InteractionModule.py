import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
from models.DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer

class InteractionModule(nn.Module):
    def __init__(self, opt, num_layer_routing=3, num_cells=4, path_hid=128):
        super(InteractionModule, self).__init__()
        self.opt = opt
        self.num_cells = num_cells = 4
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(opt, num_cells, num_cells)
        self.dynamic_itr_l1 = DynamicInteraction_Layer(opt, num_cells, num_cells)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(opt, num_cells, 1)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(opt.embed_size)


    def calSim_t2i(self, ref_wrds, img, wrd, stc_lens):
        ''' ref_wrds--(n_img, n_stc, n_wrd, d), 
            img -- (n_img, d)
        '''
        n_img, n_stc, n_wrd, d = ref_wrds.size()
        ref_wrds = (self.bn(ref_wrds.view(n_img*n_stc*n_wrd, d))).view(n_img, n_stc, n_wrd, d)
        ref_wrds = ref_wrds.mean(2) + ref_wrds.max(2)[0]
        ref_wrds = F.normalize(ref_wrds, dim=-1)    # (n_img, n_stc, d)
        img = img.unsqueeze(1) # (n_img, 1, d)
        sim = (img * ref_wrds).sum(-1)
        return sim

    def calScores(self, aggr_res, rgn, img, wrd, stc, stc_lens):
        assert len(aggr_res) == 1
        aggr_res = aggr_res[0]
        if self.opt.direction == 'i2t':
            scores = self.calSim_i2t(aggr_res, stc, rgn)
        else:
            scores = self.calSim_t2i(aggr_res, img, wrd, stc_lens)      
        return scores

    def forward(self, rgn, img, wrd, stc, stc_lens):
        '''
        rgn : torch.Size([32, 36, 256]) -> torch.Size([bsz, 4 , 1024]) -> torch.Size([bsz, 4 , 512])
        img : torch.Size([32, 256])
        wrd : torch.Size([64, 32, 256]) -> torch.Size([bsz, max_len , dim]) -> torch.Size([bsz, max_len ,512])
        stc : torch.Size([64, 256])
        stc_lens : 64* 1
        '''

        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(rgn, img, wrd, stc, stc_lens)
        pairs_emb_lst, paths_l1 = self.dynamic_itr_l1(pairs_emb_lst, rgn, img, wrd, stc, stc_lens)
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, rgn, img, wrd, stc, stc_lens)
        score = self.calScores(pairs_emb_lst, rgn, img, wrd, stc, stc_lens)

        n_img, n_stc = paths_l2.size()[:2]
        if self.opt.direction == 'i2t':
            paths_l0 = paths_l0.contiguous().view(n_img, -1).unsqueeze(1).expand(-1, n_stc, -1)
            paths_l1 = paths_l1.view(n_img, n_stc, -1)
            paths_l2 = paths_l2.view(n_img, n_stc, -1)
            paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)
            paths = paths.mean(dim=1) # (n_img, total_paths)
        else:
            paths_l0 = paths_l0.contiguous().view(n_stc, -1).unsqueeze(0).expand(n_img, -1, -1)
            paths_l1 = paths_l1.view(n_img, n_stc, -1)
            paths_l2 = paths_l2.view(n_img, n_stc, -1)
            paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)
            paths = paths.mean(dim=0) # (n_stc, total_paths)
            
        paths = self.path_mapping(paths)
        paths = F.normalize(paths, dim=-1)
        sim_paths = paths.matmul(paths.t())

        if self.training:
            return score, sim_paths
        else:
            return score