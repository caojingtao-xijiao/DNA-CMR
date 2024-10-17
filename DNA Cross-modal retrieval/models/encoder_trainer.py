import torch
import torch.nn as nn
class EncoderTrainer(nn.Module):
    def __init__(self, encoder,predictor):
        super(EncoderTrainer, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.predictor_trainable(self.predictor,False)

    def forward(self,X_pairs):
        X1, X2 = X_pairs[:,0,:],X_pairs[:,1,:]
        S1 = self.encoder(X1)#(b,80,4)
        S2 = self.encoder(X2)#(b,80,4)
        S_pairs = torch.stack([S1, S2], dim=1)#(b,2,80,4)
        S_pairs = torch.transpose(S_pairs,1,3)#(b,4,80,2)
        y = self.predictor(S_pairs)
        return y,S1,S2

    def predictor_trainable(self,model,flag):
        for param in model.parameters():
            param.requires_grad = flag









