from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMethod

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)



class MaxEnt(BaseMethod):
    """ implements maximum entropy s.t. invariance constraints """

    def __init__(self, cfg):
        """ init additional BN used after head """
        super().__init__(cfg)
        #self.bn_last = nn.BatchNorm1d(cfg.emb)
        self.norm = L2Norm()
        self.dual_var = cfg.dual_init
        self.dual_lr = cfg.dual_lr
        self.epsilon = cfg.epsilon
        self.k = cfg.entropy_k
        self.counter = 0
        self.constraint = 0
    
    def koza_leon(self, x):
        alignment = x @ x.T * (torch.eye(x.shape[0], device=x.device)==0) # eliminate diag elements
        max_alignment, _ = torch.sort(alignment, dim=0, descending=True)
        return -(2*(1-max_alignment[self.k,:])).log().mean()
    
    def align_loss(self, x, y):
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def forward(self, samples):
        bs = len(samples[0])
        h = self.head(self.model(torch.cat(samples).cuda(non_blocking=True)))
        h = self.norm(h)
        align_loss = 0
        entropy_loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs : (i + 1) * bs]
                x1 = h[j * bs : (j + 1) * bs]
                align_loss += self.align_loss(x0, x1)
                entropy_loss += (self.koza_leon(x0)+self.koza_leon(x1))/2
        #align_loss, entropy_loss =  align_loss/self.num_pairs , entropy_loss/self.num_pairs
        lagrangian = entropy_loss + self.dual_var*align_loss
        #print(align_loss)
        return lagrangian, entropy_loss.item(), align_loss.item()

    def update_slack(self, constraint_eval):
        self.counter += 1
        self.constraint += constraint_eval
    
    def update_dual(self):
        self.constraint = self.constraint/self.counter
        slack = self.constraint-self.epsilon
        self.dual_var = max(0, self.dual_var+self.dual_lr*slack)
        self.counter = 0
        self.constraint = 0

