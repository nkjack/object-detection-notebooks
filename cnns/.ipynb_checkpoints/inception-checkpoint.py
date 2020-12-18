import torch
import itertools as it
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_path_a: list, c_path_b: list, c_path_c: list, c_path_d: list):
        super().__init__()
        
        # conv 1x1 + 1(s)
        path_a = [
            nn.Conv2d(c_in, c_path_a[0], kernel_size=1, stride=1),
            nn.ReLU(),
        ]
        
        # conv 1x1 + 1(s) -> conv 3x3 + 1(s)
        path_b = [
            nn.Conv2d(c_in, c_path_b[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_path_b[0], c_path_b[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        
        # conv 1x1 + 1(s) -> conv 5x5 + 1(s)
        path_c = [
            nn.Conv2d(c_in, c_path_c[0], kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(c_path_c[0], c_path_c[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        ]
        
        # MaxPool 3x3 + 1(s) -> conv 1+1 + 1(s)
        path_d = [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_path_d[0], kernel_size=1, stride=1),
            nn.ReLU(),
        ]
    
        self.path_a = nn.Sequential(*path_a)
        self.path_b = nn.Sequential(*path_b)
        self.path_c = nn.Sequential(*path_c)
        self.path_d = nn.Sequential(*path_d)

        # ========================

    def forward(self, x):
        out_a = self.path_a(x)
        out_b = self.path_b(x)
        out_c = self.path_c(x)
        out_d = self.path_d(x)
        out = torch.cat([out_a,out_b,out_c,out_d], dim=1)
        return out


class InceptionNet(nn.Module):
    def __init__(self, in_size, out_classes: int):
        super().__init__()
#         assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes

        layers = [
            nn.Conv2d(in_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(192,[64],[96,128],[16,32],[32]),
            InceptionBlock(256, [128],[128,192],[32,96],[64]),
            #
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(480, [192],[96,208],[16,48],[64]), # here can go out to fc
            InceptionBlock(512, [160],[112,224],[24,64],[64]),
            InceptionBlock(512, [128],[128,256],[24,64],[64]),
            InceptionBlock(512, [112],[144,288],[32,64],[64]),
            InceptionBlock(528, [256],[160,320],[32,128],[128]),
            #
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(832, [256],[160,320],[32,128],[128]),
            InceptionBlock(832, [384],[192,384],[48,128],[128]),            
            
            nn.AvgPool2d(kernel_size=7, stride=1),
#             nn.AvgPool2d(kernel_size=5, stride=1),
#             nn.Conv2d(cin, cout, kernel_size=1, stride=1),
        ]
        
        fc_layers = [
            nn.Dropout(0.4),
            nn.Linear(1024, self.out_classes),
            nn.Softmax(dim=-1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Sequential(*fc_layers)


    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.fc(features)
        out = class_scores
        # ========================
        return out
        

class InceptionNetMini(nn.Module):
    def __init__(self, in_size, out_classes: int):
        super().__init__()
#         assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        
        layers = [
            nn.Conv2d(in_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(192,[64],[96,128],[16,32],[32]),
            InceptionBlock(256, [128],[128,192],[32,96],[64]),
            #
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(480, [192],[96,208],[16,48],[64]),             
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU()
            
        ]
        
        fc_layers = [
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_classes),
            nn.Softmax(dim=-1)
        ]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Sequential(*fc_layers)


    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.fc(features)
        out = class_scores
        # ========================
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
        