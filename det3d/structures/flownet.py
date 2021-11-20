import torch

def MLP(channels, batch_norm=True):
    module_list = []
    for i in range(1, len(channels)):
        module = torch.nn.Linear(channels[i-1], channels[i])
        module = torch.nn.Sequential(module, torch.nn.ReLU(),
                                     torch.nn.BatchNorm1d(channels[i]))
        module_list.append(module)
    return torch.nn.Sequential(*module_list)

class TFlowNet(torch.nn.Module):
    """ Predict flow from (x,y,z,t)
    Args:
        points_txyz (torch.tensor, [N, 4]): temporal point cloud

    Returns:
        velo (torch.tensor, [N, 3]): velocity of each point

    """
    def __init__(self,
                 channels = [(4, 128), (128, 128), (128, 128), (128, 128),
                             (128, 128), (128, 128), (128, 3)],
                 **kwargs):
        super(TFlowNet, self).__init__(**kwargs)
        self.layers = []
        for i, channel in enumerate(channels):
            self.layers.append(MLP(channel))
            self.__setattr__(f'mlp{i}', self.layers[-1])

    def forward(self, points_txyz):
        points = points_txyz
        for layer in self.layers:
            next_points = layer(points)
            if next_points.shape[-1] == points.shape[-1]:
                next_points += points
            points = next_points

        return next_points

if __name__ == '__main__':
    net = TFlowNet()
    points_txyz = torch.randn(100, 4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    for itr in range(10000):
        optimizer.zero_grad()
        v = net(points_txyz)
        loss = (points_txyz[:, :3] + v).square().sum()
        loss.backward()
        optimizer.step()
        print(f'loss={loss:.4f}')
    
