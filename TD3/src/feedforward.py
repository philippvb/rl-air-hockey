import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """Implementation of a feedforward network with fully connected layers
    """
    def __init__(self, dimensions, actor=True):
        """Initializes the network

        Args:
            dimensions (list[int]): The sizes of the layers
            actor (bool, optional): Wether the network should output a distribution between -1 and 1 (needed for actor). Defaults to True.
        """
        super(FeedForwardNetwork, self).__init__()
        layers=[]
        for i in range(len(dimensions)-1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(nn.ReLU())

        self.layers = layers[:-1]
        if actor:
            self.layers.append(nn.Tanh()) # replace the last Relu with Tanh

        self.layers_forward = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.layers_forward(x)
