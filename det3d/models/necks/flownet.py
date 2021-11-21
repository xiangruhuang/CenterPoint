import torch
from ..registry import NECKS 

def MLP(channels, batch_norm=True, relu=True, std=1):
    module_list = []
    for i in range(1, len(channels)):
        module = torch.nn.Linear(channels[i-1], channels[i])
        #torch.nn.init.normal_(module.weight, std=std)
        torch.nn.init.zeros_(module.bias)
        module_list.append(module)
        if relu:
            module_list.append(torch.nn.ReLU())
        if batch_norm:
            module_list.append(torch.nn.BatchNorm1d(channels[i]))
    return torch.nn.Sequential(*module_list)

@NECKS.register_module
class TFlowNet(torch.nn.Module):
    """ Predict (temporal) flow from any point (x,y,z,t)
    Args:
        points_xyzt (torch.tensor, [N, 4]): temporal point cloud

    Returns:
        velo (torch.tensor, [N, 3]): velocity of each point

    """
    def __init__(self,
                 channels = [(4, 128, 128, 128, 128, 128),
                             (128, 128, 128), (128, 3)],
                 **kwargs):
        super(TFlowNet, self).__init__(**kwargs)
        self.layers = []
        for i, channel in enumerate(channels):
            if i == len(channels) - 1:
                layer = MLP(channel, batch_norm=False, relu=False, std=1e-2)
            else:
                layer = MLP(channel, batch_norm=True, std=1e-2)
            self.layers.append(layer)
            self.__setattr__(f'mlp{i}', self.layers[-1])

    def forward(self, points_xyzt):
        points = points_xyzt
        for layer in self.layers:
            next_points = layer(points)
            if next_points.shape[-1] == points.shape[-1]:
                next_points += points
            points = next_points

        return next_points
