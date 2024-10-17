import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from params import share_param


class Encoder(nn.Module):
    def __init__(self,
                 feature_len=share_param.feature_len,
                 DNA_len=share_param.DNA_len,
                 ):
        super(Encoder, self).__init__()
        self.feature_len = feature_len
        self.DNA_len = DNA_len

        self.encoder_model = nn.Sequential(
            nn.Linear(in_features=feature_len, out_features=feature_len*2),
            nn.ReLU(),
            nn.Linear(in_features=feature_len*2, out_features=feature_len*4),
            nn.ReLU(),
            nn.Linear(in_features=feature_len*4, out_features=feature_len*2),
            nn.ReLU(),
        )
        self.last_linear = nn.Linear(in_features=self.feature_len*2, out_features=self.DNA_len*4)
        self.softmax = nn.Softmax(dim=-1)

        self.rea = Rearrange('b (h w) -> b h w ',h=DNA_len,w=4)
        # self.initialize()
        self.bases = np.array(list("ATCG"))
    # x --> (b,512) --> (b,80,4)
    def forward(self,x):
        # x = x / x.norm(dim=-1, keepdim=True)
        x = self.encoder_model(x)
        x = self.last_linear(x)
        x = self.rea(x)
        out = self.softmax(x)
        return out

    #feature --> DNA序列
    def feature_to_seq(self,x):
        out = self.forward(x)
        return self.prob_to_seq(out.cpu())
    #概率or独热吗--->DNA序列
    #(batch,DNA_seq,4)--->(batch,1)
    def prob_to_seq(self,onehot):
        id = np.array(torch.argmax(onehot, dim=-1).cpu()).reshape(-1, onehot.shape[1])
        lst = []
        for i in self.bases[id]:
            lst.append(''.join(list(i)))
        return np.array(lst)

