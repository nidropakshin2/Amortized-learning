import torch
import torch.nn as nn
from sfmpe.core.distributions import Distribution
from sfmpe.flow.path import Path

# TODO: посмотреть влияет ли это на что-нибудь
class FlowModel(nn.Module):
# class FlowModel:
    def __init__(self, 
                 velocity_model,
                 init_dist: Distribution,
                 path: Path):
        super().__init__()
        self.velocity_model = velocity_model
        self.init_dist      = init_dist
        self.path           = path 

    def velocity(self, t, theta, x):
        return self.velocity_model(t, theta, x)


