import torch
import itertools as it
import torch.nn as nn


class DWConv(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
        super(DWConv, self).__init__(
            nn.Conv2d(c_in, c_in, kernel_size, stride, padding=1, groups=groups),
            nn.BatchNorm2d(c_in),
#             nn.ReLU6(inplace=True),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )


        
class MobileNetV1(nn.Module):
    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        
        layers_features = [
            #
            nn.Conv2d(in_size[0], 32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #
            DWConv(32,64,3,1,32),
            DWConv(64,128,3,2,64),
            DWConv(128,128,3,1,128),
            DWConv(128,256,3,2,128),
            #
            DWConv(256,256,3,1,256),
            DWConv(256,512,3,2,256),
            #
            DWConv(512,512,3,1,512),
            DWConv(512,512,3,1,512),
            DWConv(512,512,3,1,512),
            DWConv(512,512,3,1,512),
            DWConv(512,512,3,1,512),
            #
            DWConv(512,1024,3,2,512),
            DWConv(1024,1024,3,1,1024),
            nn.AvgPool2d(7, stride=1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers_features)
        
        t = torch.rand(1,3,224,224)
        num_neurons = self.feature_extractor(t).view(1,-1).shape[1]
#         print(self.feature_extractor(t).shape)
        
        layers_classifier = [
            nn.Linear(num_neurons, 10),
            nn.Softmax(dim=-1)
        ]
        
        self.classifier = nn.Sequential(*layers_classifier)
        
        # weight initialization from torchvision implementation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        out = class_scores
        return out

    

# from https://github.com/pytorch/vision/blob/898802fe41fa060328649ae0a60cbd72110b4633/torchvision/models/mobilenet.py
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )
        

# from https://github.com/pytorch/vision/blob/898802fe41fa060328649ae0a60cbd72110b4633/torchvision/models/mobilenet.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
    

class MobileNetV2(nn.Module):
    def __init__(self, in_size, out_classes: int):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        
        layers_features = [
            #
            nn.Conv2d(in_size[0], 32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #
            InvertedResidual(32, 16, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d),
            # 
            InvertedResidual(16, 24, stride=2, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(24, 24, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            InvertedResidual(24, 32, stride=2, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(32, 32, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(32, 32, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            InvertedResidual(32, 64, stride=2, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            InvertedResidual(64, 96, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(96, 96, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(96, 96, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            InvertedResidual(96, 160, stride=2, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(160, 160, stride=1,expand_ratio=6, norm_layer=nn.BatchNorm2d),
            InvertedResidual(160, 160, stride=1,expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            InvertedResidual(160, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d),
            #
            nn.Conv2d(320, 1280, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
#             #
            nn.AvgPool2d(7, stride=1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers_features)
        
        t = torch.rand(1,3,224,224)
        num_neurons = self.feature_extractor(t).view(1,-1).shape[1]
#         print(self.feature_extractor(t).shape)
        
        layers_classifier = [
            nn.Linear(num_neurons, 10),
            nn.Softmax(dim=-1)
        ]
        
        self.classifier = nn.Sequential(*layers_classifier)
        
        # weight initialization from torchvision implementation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
# def weights_init(m):
#     classname = m.__class__.__name__
#     if hasattr(m, 'weight') and (classname.find('Conv') != -1):
#         nn.init.normal_(m.weight.data, 0.0, 0.01)
#     elif hasattr(m, 'weight') and (classname.find('Linear') != -1):
#         nn.init.normal_(m.weight.data, 0.0, 0.01)
#         nn.init.constant_(m.bias.data, 1)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
        