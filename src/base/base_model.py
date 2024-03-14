from torch import nn


class BaseModel(nn.Module):
    def forward(self):
        NotImplementedError
