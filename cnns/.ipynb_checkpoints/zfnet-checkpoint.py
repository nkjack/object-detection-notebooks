# some parts of the code from https://hackmd.io/@bouteille/ByaTE80BI

import torch
import itertools as it
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        
        layers_features = [
            nn.Conv2d(in_size[0], 96, kernel_size=7, stride=2,padding=1),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2,padding=0, bias=True),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True),
        ]
        
        layers_classifier = [
            nn.Linear(9216, 4096),
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
        
        
        self.feature_outputs = [0]*len(self.feature_extractor)
        self.switch_indices = dict()
        self.sizes = dict()
        
        # deconv part
        self.deconv_pool5 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=0)
        self.deconv_act5 = nn.ReLU()
        self.deconv_conv5 = nn.ConvTranspose2d(256,384,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.deconv_act4 = nn.ReLU()
        self.deconv_conv4 = nn.ConvTranspose2d(384,384,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.deconv_act3 = nn.ReLU()
        self.deconv_conv3 = nn.ConvTranspose2d(384,256,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.deconv_pool2 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=1)
        self.deconv_act2 = nn.ReLU()
        self.deconv_conv2 = nn.ConvTranspose2d(256,96,kernel_size=5,stride=2,padding=0,bias=False)
        
        self.deconv_pool1 = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=1)
        self.deconv_act1 = nn.ReLU()
        self.deconv_conv1 = nn.ConvTranspose2d(96,3,kernel_size=7,stride=2,padding=1,bias=False)
        
    def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(features.size(0), -1)
#         class_scores = self.classifier(features)
#         out = class_scores
        for i, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)    
                self.switch_indices[i] = indices
            else:
                x = layer(x)
            self.feature_outputs[i] = x
            
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
    
    def forward_deconv(self, x, layer):
        assert layer >= 1 and layer <= 5
    
        x = self.deconv_pool5(x, self.switch_indices[14])
        x = self.deconv_act5(x)
        x = self.deconv_conv5(x)
        
        if layer == 1: return x
        
        x = self.deconv_act4(x)
        x = self.deconv_conv4(x)
        
        if layer == 2: return x
        
        x = self.deconv_act3(x)
        x = self.deconv_conv3(x)
        
        if layer == 3: return x
        
        x = self.deconv_pool2(x, self.switch_indices[7], output_size=self.feature_outputs[6].shape[-2:])
        x = self.deconv_act2(x)
        x = self.deconv_conv2(x)
     
        if layer == 4: return x
        
        x = self.deconv_pool1(x, self.switch_indices[3], output_size=self.feature_outputs[2].shape[-2:])
        x = self.deconv_act1(x)
        x = self.deconv_conv1(x)
        
        if layer == 5:
            return x
        

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
        