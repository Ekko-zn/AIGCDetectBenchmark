import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL
        #print (features.size(),features.transpose(1,2).size())
        #G = torch.bmm(features, features.transpose(1,2))  # compute the gram product
        a= features.transpose(1,2)
        G = torch.bmm(features, a)
        #print (G.size)
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        G=G.unsqueeze(1)
        return G.div(b* c * d)
class ScaleLayer(nn.Module):

   def __init__(self, init_value=1):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.gram = GramMatrix()
        self.scale=ScaleLayer()
        self.fcnewr = nn.Sequential(nn.Linear(704, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(512, 1)
                                   )


        self.conv_interi_0 = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))

        self.conv_inter0_0 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))



        self.conv_inter1_0 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))



        self.conv_inter2_0 = nn.Sequential(nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))

        self.conv_inter3_0 = nn.Sequential(nn.Conv2d(128,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))


        self.conv_inter4_0 = nn.Sequential(nn.Conv2d(256,32, kernel_size=3, stride=1, padding=1,
                               bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))



        self.gi_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.gi_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.g0_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g0_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.g_fc1r = nn.Sequential(nn.Conv2d(1,16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g_fc2r = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False), nn.BatchNorm2d(32),nn.ReLU())


        self.g2_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g2_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.g3_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g3_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())

        self.g4_fc1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(16), nn.ReLU())
        self.g4_fc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1,
                               bias=False),nn.BatchNorm2d(32), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x3=x
        x = self.conv1(x3)
        x = self.bn1(x)
        x4 = self.relu(x)
        x5 = self.maxpool(x4)


        x6 = self.layer1(x5)
        x7 = self.layer2(x6)
        x8 = self.layer3(x7)
        x = self.layer4(x8)

        

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        gi=self.conv_interi_0(x3)

        gi=self.gram(gi)

        gi=self.gi_fc1(gi)
        gi=self.gi_fc2(gi)
        gi = self.avgpool(gi)
        gi = gi.view(gi.size(0), -1)

        g0=self.conv_inter0_0(x4)

        g0=self.gram(g0)

        g0=self.g0_fc1(g0)
        g0=self.g0_fc2(g0)
        g0 = self.avgpool(g0)
        g0 = g0.view(g0.size(0), -1)
        g1=self.conv_inter1_0(x5)

        g1=self.gram(g1)

        g1=self.g_fc1r(g1)
        g1=self.g_fc2r(g1)
        g1 = self.avgpool(g1)
        g1 = g1.view(g1.size(0), -1)

        g2=self.conv_inter2_0(x6)

        g2=self.gram(g2)

        g2=self.g2_fc1(g2)
        g2=self.g2_fc2(g2)
        g2 = self.avgpool(g2)
        g2 = g2.view(g2.size(0), -1)
        g3=self.conv_inter3_0(x7)

        g3=self.gram(g3)

        g3=self.g3_fc1(g3)
        g3=self.g3_fc2(g3)
        g3 = self.avgpool(g3)
        g3 = g3.view(g3.size(0), -1)
        g4=self.conv_inter4_0(x8)

        g4=self.gram(g4)

        g4=self.g4_fc1(g4)
        g4=self.g4_fc2(g4)
        g4 = self.avgpool(g4)
        g4 = g4.view(g4.size(0), -1)
        gi=self.scale(gi)
        g0=self.scale(g0)
        g1=self.scale(g1)
        g2=self.scale(g2)
        g3=self.scale(g3)
        g4=self.scale(g4)
        '''a=np.unique(x.cpu().detach().numpy())
        b=np.unique(gi.cpu().detach().numpy())
        c=np.unique(g0.cpu().detach().numpy())
        d=np.unique(g1.cpu().detach().numpy())
        e=np.unique(g2.cpu().detach().numpy())
        f=np.unique(g3.cpu().detach().numpy())
        g=np.unique(g4.cpu().detach().numpy())
        print (a,b,c,d,e,f,g)
        exit()'''

        x=torch.cat((x,gi,g0,g1,g2,g3,g4),1)
        x = self.fcnewr(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model

if __name__=='__main__':
    net = resnet18()
    x =  torch.rand(10,3,224,224)
    net(x)