import torch
import itertools as it
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        
        layers_features = [
            nn.Conv2d(in_size[0], 48, kernel_size=11, stride=4,padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, stride=1,padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]

        layers_classifier = [
            nn.Linear(21632, 4096),
            nn.Dropout(0),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0),
            nn.ReLU(),
            nn.Linear(4096, out_classes),
            nn.Softmax(dim=-1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers_features)
        self.classifier = nn.Sequential(*layers_classifier)

    def forward(self, x):
        # Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        out = class_scores
        return out

# custom weights initialization called on netG and netD        
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif hasattr(m, 'weight') and (classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.constant_(m.bias.data, 1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        