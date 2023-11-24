import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
import torchvision
import torch.nn.functional as F





class VGG(torch.nn.Module):
    def __init__(self, arch_type, pretrained, progress):
        super().__init__()
        
        self.layer1 = torch.nn.Sequential()
        self.layer2 = torch.nn.Sequential()
        self.layer3 = torch.nn.Sequential()
        self.layer4 = torch.nn.Sequential()
        self.layer5 = torch.nn.Sequential()

        if arch_type == 'vgg11':
            official_vgg = torchvision.models.vgg11(pretrained=pretrained, progress=progress)
            blocks = [  [0,2], [2,5], [5,10], [10,15], [15,20] ]
            last_idx = 20
        elif arch_type == 'vgg19':
            official_vgg = torchvision.models.vgg19(pretrained=pretrained, progress=progress)
            blocks = [  [0,4], [4,9], [9,18], [18,27], [27,36] ]
            last_idx = 36
        else:
            raise NotImplementedError
        
        
        for x in range( *blocks[0] ):
            self.layer1.add_module(str(x), official_vgg.features[x])
        for x in range( *blocks[1] ):
            self.layer2.add_module(str(x), official_vgg.features[x])
        for x in range( *blocks[2] ):
            self.layer3.add_module(str(x), official_vgg.features[x])
        for x in range( *blocks[3] ):
            self.layer4.add_module(str(x), official_vgg.features[x])
        for x in range( *blocks[4] ):
            self.layer5.add_module(str(x), official_vgg.features[x])
            
        self.max_pool = official_vgg.features[last_idx]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc1 = official_vgg.classifier[0]
        self.fc2 = official_vgg.classifier[3]
        self.fc3 = official_vgg.classifier[6]
        self.dropout = nn.Dropout()
        
        
    def forward(self, x):
        out = {}
        
        x = self.layer1(x)
        out['f0'] = x
        
        x = self.layer2(x)
        out['f1'] = x
        
        x = self.layer3(x)
        out['f2'] = x
        
        x = self.layer4(x)
        out['f3'] = x
        
        x = self.layer5(x)
        out['f4'] = x
        
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view(-1,512*7*7) 
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = F.relu(x)
        out['penultimate'] = x 
        x = self.dropout(x) 
        x = self.fc3(x)
        out['logits'] = x 

        return out










def vgg11(pretrained=False, progress=True):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG('vgg11', pretrained, progress)



def vgg19(pretrained=False, progress=True):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG('vgg19', pretrained, progress)




