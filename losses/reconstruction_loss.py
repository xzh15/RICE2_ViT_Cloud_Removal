import torch.nn as nn

class ReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self,prediction,target):
        return self.loss(prediction,target)