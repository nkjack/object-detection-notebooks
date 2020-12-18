import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        
        for idx, c_out in enumerate(self.channels):
            layers.append(nn.Conv2d(in_channels, c_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if (idx + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
            
            in_channels = c_out
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        for idx in range(len(self.channels)):
            if (idx + 1) % self.pool_every  == 0 :
                in_h, in_w = in_h/2, in_w/2
        
        in_h, in_w = int(in_h), int(in_w)
                
        for idx, h_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.append(nn.Linear(self.channels[-1]*in_h*in_w, h_dim))
            else:
                layers.append(nn.Linear(self.hidden_dims[idx-1], h_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        out = class_scores
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        main_path, shortcut_path = [], []
        shortcut_path.append(nn.Identity())
        
        c_in_tmp = in_channels
        
        for idx, c_out in enumerate(channels[:-1]):
            main_path.append(nn.Conv2d(c_in_tmp, c_out, kernel_size=kernel_sizes[idx], padding=int((kernel_sizes[idx]-1)/2)))
            if dropout > 0:
                main_path.append(nn.Dropout2d(dropout))
            if batchnorm:
                main_path.append(nn.BatchNorm2d(c_out))
            main_path.append(nn.ReLU())
        
            c_in_tmp = c_out
        
        main_path.append(nn.Conv2d(c_in_tmp, channels[-1],kernel_size=kernel_sizes[-1], padding=int((kernel_sizes[-1]-1)/2)))
        
        if channels[-1] != in_channels:
            shortcut_path.append(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))
                
        self.main_path, self.shortcut_path = nn.Sequential(*main_path), nn.Sequential(*shortcut_path)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        short = self.shortcut_path(x)
        out += short
        out = nn.functional.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        
        repeat = int(len(self.channels)/self.pool_every)
        for idx in range(repeat):
            layers.append(ResidualBlock(in_channels, 
                                        self.channels[idx*self.pool_every:(idx+1)*self.pool_every], 
                                        [3]*self.pool_every, 
                                        batchnorm=True))
            
            in_channels = self.channels[(idx+1)*self.pool_every - 1]
            layers.append(nn.MaxPool2d(kernel_size=2))
            
        if len(self.channels) % self.pool_every != 0:
            layers.append(ResidualBlock(in_channels, self.channels[repeat*self.pool_every:], [3]*(len(self.channels)-repeat*self.pool_every)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class ResNetClassifierCIFAR10(nn.Module):
    def __init__(self, in_size, out_classes):
        super().__init__()
        
        self.in_size = in_size
        self.out_classes = out_classes
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, [16,16], [3,3],  batchnorm=True),
            ResidualBlock(16, [16,16], [3,3],  batchnorm=True),
            ResidualBlock(16, [16,16], [3,3],  batchnorm=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, [32,32], [3,3],  batchnorm=True),
            ResidualBlock(32, [32,32], [3,3],  batchnorm=True),
            ResidualBlock(32, [32,32], [3,3],  batchnorm=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, [64,64], [3,3],  batchnorm=True),
            ResidualBlock(64, [64,64], [3,3],  batchnorm=True),
            ResidualBlock(64, [64,64], [3,3],  batchnorm=True),
            nn.AvgPool2d(kernel_size=8, stride=1),
        ]
        
        fc_layers = [
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        out = class_scores
        # ========================
        return out