import torchvision.models as models
import torch
import torch.nn as nn

class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif kwargs['name'] == 'resnet34':
            resnet = models.resnet34(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, resnet.conv1.out_channels,
                                 kernel_size=resnet.conv1.kernel_size,
                                 stride=resnet.conv1.stride,
                                 padding=resnet.conv1.padding,
                                 bias=resnet.conv1.bias)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size = 512, projection_size = 128):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
