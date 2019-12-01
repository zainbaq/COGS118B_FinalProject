import torch

class AlexNet(torch):

    self.model = torch.hub.load(
        'pytorch/vision:v0.4.2', 
        'alexnet',
        pretrained=True
        )
