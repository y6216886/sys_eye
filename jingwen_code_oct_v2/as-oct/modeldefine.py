import torch.nn as nn
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import random
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F


def initweights(m):
    orthogonal_flag = False
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2. / n))

            # orthogonal initialize
            """Reference:
            [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear neural networks." arXiv preprint arXiv:1312.6120 (2013)."""
            if orthogonal_flag:
                weight_shape = layer.weight.data.cpu().numpy().shape
                u, _, v = np.linalg.svd(layer.weight.data.cpu().numpy(), full_matrices=False)
                flat_shape = (weight_shape[0], np.prod(weight_shape[1:]))
                q = u if u.shape == flat_shape else v
                q = q.reshape(weight_shape)
                layer.weight.data.copy_(torch.Tensor(q))

        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

############################################
#                   ResNet                 #
############################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(initweights)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNetImageNet(nn.Module):

    def __init__(self, opt, num_classes=1000,retrain=None):
        super(ResNetImageNet, self).__init__()
        self.opt = opt
        self.depth = opt.depth
        # self.num_classes = opt.num_classes
        self.num_classes = opt.nClasses
        self.sigmoid = nn.Sigmoid()
        fc = nn.Linear(512, self.num_classes)  ##resnet50 2048     resnet18  512
        fc.bias.data.zero_()
        if retrain:
            if isinstance(retrain, nn.DataParallel):
                self.conv1 = retrain.module.conv1
                self.bn1 = retrain.module.bn1
                self.relu = retrain.module.relu
                self.maxpool = retrain.module.maxpool
                self.layer1 = retrain.module.layer1
                self.layer2 = retrain.module.layer2
                self.layer3 = retrain.module.layer3
                self.layer4 = retrain.module.layer4
                self.avgpool = retrain.module.avgpool
                self.fc = retrain.module.fc
                # self.model = retrain.module.model
            else:
                self.conv1 = retrain.conv1
                self.bn1 = retrain.bn1
                self.relu = retrain.relu
                self.maxpool = retrain.maxpool
                self.layer1 = retrain.layer1
                self.layer2 = retrain.layer2
                self.layer3 = retrain.layer3
                self.layer4 = retrain.layer4
                self.avgpool = retrain.avgpool
                self.fc = retrain.fc
                # self.model = retrain.model
            return
        if self.depth == 18:
            if self.opt.pretrain:
                pretrainmodel = models.resnet18(pretrained=True)
                self.conv1 = pretrainmodel.conv1
                self.bn1 = pretrainmodel.bn1
                self.relu = pretrainmodel.relu
                self.maxpool = pretrainmodel.maxpool
                self.layer1 = pretrainmodel.layer1
                self.layer2 = pretrainmodel.layer2
                self.layer3 = pretrainmodel.layer3
                self.layer4 = pretrainmodel.layer4
                self.avgpool = nn.AvgPool2d(16, stride=1)
                self.fc = fc
            else:
               self.model = ResNet(BasicBlock, [2, 2, 2, 2], self.num_classes)
        elif self.depth == 34:
            if self.opt.pretrain:
                pretrainmodel = models.resnet34(pretrained=True)
                self.conv1 = pretrainmodel.conv1
                self.bn1 = pretrainmodel.bn1
                self.relu = pretrainmodel.relu
                self.maxpool = pretrainmodel.maxpool
                self.layer1 = pretrainmodel.layer1
                self.layer2 = pretrainmodel.layer2
                self.layer3 = pretrainmodel.layer3
                self.layer4 = pretrainmodel.layer4
                self.avgpool = nn.AvgPool2d(16, stride=1)
                self.fc = fc
            else:
               self.model = ResNet(BasicBlock, [3, 4, 6, 3], self.num_classes)
        elif self.depth == 50:
            if self.opt.pretrain:
                pretrainmodel = models.resnet50(pretrained=True)
                self.conv1 = pretrainmodel.conv1
                self.bn1 = pretrainmodel.bn1
                self.relu = pretrainmodel.relu
                self.maxpool = pretrainmodel.maxpool
                self.layer1 = pretrainmodel.layer1
                self.layer2 = pretrainmodel.layer2
                self.layer3 = pretrainmodel.layer3
                self.layer4 = pretrainmodel.layer4
                self.avgpool = nn.AvgPool2d(16, stride=1)  #imgsize=512
                # self.avgpool = nn.AvgPool2d(8, stride=1)   #imgsize=256
                self.fc = fc
            else:
               self.model = ResNet(Bottleneck, [3, 4, 6, 3], self.num_classes)
        elif self.depth == 101:
            if self.opt.pretrain:
                pretrainmodel = models.resnet101(pretrained=True)
                self.conv1 = pretrainmodel.conv1
                self.bn1 = pretrainmodel.bn1
                self.relu = pretrainmodel.relu
                self.maxpool = pretrainmodel.maxpool
                self.layer1 = pretrainmodel.layer1
                self.layer2 = pretrainmodel.layer2
                self.layer3 = pretrainmodel.layer3
                self.layer4 = pretrainmodel.layer4
                self.avgpool = nn.AvgPool2d(16, stride=1)
                self.fc = fc
            else:
               self.model = ResNet(Bottleneck, [3, 4, 23, 3], self.num_classes)
        elif self.depth == 152:
            if self.opt.pretrain:
                pretrainmodel = models.resnet152(pretrained=True)
                self.conv1 = pretrainmodel.conv1
                self.bn1 = pretrainmodel.bn1
                self.relu = pretrainmodel.relu
                self.maxpool = pretrainmodel.maxpool
                self.layer1 = pretrainmodel.layer1
                self.layer2 = pretrainmodel.layer2
                self.layer3 = pretrainmodel.layer3
                self.layer4 = pretrainmodel.layer4
                self.avgpool = nn.AvgPool2d(16, stride=1)
                self.fc = fc
            else:
               self.model = ResNet(Bottleneck, [3, 8, 36, 3], self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print ("avgpool", x.size ())
        x = x.view(x.size(0), -1)
        # print("fc", x.size())
        x = self.fc(x)
        # print("fc", x)
        if self.opt.trainingType == 'onevsall':
            x = self.sigmoid(x)
        # print("sig", x)
        return x


############################################
#                 DenseNet                 #
############################################

class DenseNetImageNet(nn.Module):

    def __init__(self, opt, num_classes=1000, retrain=None):
        super(DenseNetImageNet, self).__init__()
        self.opt = opt
        self.depth = opt.depth
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.trainingType = self.opt.trainingType
        if retrain:
            if isinstance(retrain, nn.DataParallel):
                self.features = retrain.module.features
                self.classifier = retrain.module.classifier
            else:
                self.features = retrain.features
                self.classifier = retrain.classifier
            return
        if self.depth == 121:
            pretrainmodel = models.densenet121(pretrained=True)
        elif self.depth == 161:
            pretrainmodel = models.densenet161(pretrained=True)
        elif self.depth == 169:
            pretrainmodel = models.densenet169(pretrained=True)
        elif self.depth == 201:
            pretrainmodel = models.densenet201(pretrained=True)
        self.features = pretrainmodel.features
        num_features = pretrainmodel.classifier.in_features
        self.classifier = nn.Linear(num_features, self.num_classes)
        self.classifier.bias.data.zero_() 

    def forward(self, x):
        features = self.features(x)
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=16, stride=1).view(features.size(0), -1)
        x = self.classifier(x)
        if self.trainingType == 'onevsall':
            x = self.sigmoid(x)
        return x    

############################################
#                 Inception                #
############################################

class Inception3ImageNet(nn.Module):

    def __init__(self, opt, num_classes=1000, retrain=None):
        super(Inception3ImageNet, self).__init__()
        self.opt = opt
        self.depth = opt.depth
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        if retrain:
            if isinstance(retrain, nn.DataParallel):
                # self.model = retrain.module.model
                self.aux_logits = retrain.module.aux_logits
                self.transform_input = retrain.module.transform_input
                self.Conv2d_1a_3x3 = retrain.module.Conv2d_1a_3x3
                self.Conv2d_2a_3x3 = retrain.module.Conv2d_2a_3x3
                self.Conv2d_2b_3x3 = retrain.module.Conv2d_2b_3x3
                self.Conv2d_3b_1x1 = retrain.module.Conv2d_3b_1x1
                self.Conv2d_4a_3x3 = retrain.module.Conv2d_4a_3x3
                self.Mixed_5b = retrain.module.Mixed_5b
                self.Mixed_5c = retrain.module.Mixed_5c
                self.Mixed_5d = retrain.module.Mixed_5d
                self.Mixed_6a = retrain.module.Mixed_6a
                self.Mixed_6b = retrain.module.Mixed_6b
                self.Mixed_6c = retrain.module.Mixed_6c
                self.Mixed_6d = retrain.module.Mixed_6d
                self.Mixed_6e = retrain.module.Mixed_6e
                if self.aux_logits:
                    self.AuxLogits = retrain.module.AuxLogits
                self.Mixed_7a = retrain.module.Mixed_7a
                self.Mixed_7b = retrain.module.Mixed_7b
                self.Mixed_7c = retrain.module.Mixed_7c
                self.fc = retrain.module.fc
            else:
                # self.model = retrain.model
                self.aux_logits = retrain.aux_logits
                self.transform_input = retrain.transform_input
                self.Conv2d_1a_3x3 = retrain.Conv2d_1a_3x3
                self.Conv2d_2a_3x3 = retrain.Conv2d_2a_3x3
                self.Conv2d_2b_3x3 = retrain.Conv2d_2b_3x3
                self.Conv2d_3b_1x1 = retrain.Conv2d_3b_1x1
                self.Conv2d_4a_3x3 = retrain.Conv2d_4a_3x3
                self.Mixed_5b = retrain.Mixed_5b
                self.Mixed_5c = retrain.Mixed_5c
                self.Mixed_5d = retrain.Mixed_5d
                self.Mixed_6a = retrain.Mixed_6a
                self.Mixed_6b = retrain.Mixed_6b
                self.Mixed_6c = retrain.Mixed_6c
                self.Mixed_6d = retrain.Mixed_6d
                self.Mixed_6e = retrain.Mixed_6e
                if self.aux_logits:
                    self.AuxLogits = retrain.AuxLogits
                self.Mixed_7a = retrain.Mixed_7a
                self.Mixed_7b = retrain.Mixed_7b
                self.Mixed_7c = retrain.Mixed_7c
                self.fc = retrain.fc
            return

        pretrainmodel = models.inception_v3(pretrained=True)
        self.aux_logits = pretrainmodel.aux_logits
        self.transform_input = pretrainmodel.transform_input
        self.Conv2d_1a_3x3 = pretrainmodel.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = pretrainmodel.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = pretrainmodel.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = pretrainmodel.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = pretrainmodel.Conv2d_4a_3x3
        self.Mixed_5b = pretrainmodel.Mixed_5b
        self.Mixed_5c = pretrainmodel.Mixed_5c
        self.Mixed_5d = pretrainmodel.Mixed_5d
        self.Mixed_6a = pretrainmodel.Mixed_6a
        self.Mixed_6b = pretrainmodel.Mixed_6b
        self.Mixed_6c = pretrainmodel.Mixed_6c
        self.Mixed_6d = pretrainmodel.Mixed_6d
        self.Mixed_6e = pretrainmodel.Mixed_6e
        if self.aux_logits:
            self.AuxLogits = pretrainmodel.AuxLogits
        self.Mixed_7a = pretrainmodel.Mixed_7a
        self.Mixed_7b = pretrainmodel.Mixed_7b
        self.Mixed_7c = pretrainmodel.Mixed_7c
        self.fc = nn.Linear(2048, self.num_classes)
        self.fc.bias.data.zero_()
        
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)   
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)  
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)  
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)   
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)  
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=14)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.opt.trainingType == 'onevsall':
            x = self.sigmoid(x)
        return x
        # if self.training:
        #     x, _ = self.model(x)
        # else:
        #     x = self.model(x)
        # if self.opt.trainingType == 'onevsall':
        #     x = self.sigmoid(x)
        # return x    


############################################
#                 AlexNet                  #
############################################
# class AlexNetImageNet(nn.Module):

#     def __init__(self, opt, num_classes=1000):
#         super(AlexNetImageNet, self).__init__()
#         self.opt = opt
#         self.depth = opt.depth
#         self.num_classes = num_classes
#         self.sigmoid = nn.Sigmoid()
#         self.model = models.alexnet(pretrained=True)
#         num_ftrs = list(self.model.classifier.children())[-1].in_features
#         self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1], nn.Linear(num_ftrs, self.num_classes))
#         #list(self.model.classifier.children())[-1].bias.data.zero_()

#     def forward(self, x):
#         x = self.model(x)
#         if self.opt.trainingType == 'onevsall':
#             x = self.sigmoid(x)
#         return x    


############################################
#                 VGG                      #
############################################
class VGGImageNet(nn.Module):

    def __init__(self, opt, num_classes=1000, retrain=None):
        super(VGGImageNet, self).__init__()
        self.opt = opt
        self.depth = opt.depth
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        if retrain:
            if isinstance(retrain, nn.DataParallel):
                # self.model = retrain.module.model
                self.features = retrain.module.features
                self.classifier = retrain.module.classifier
            else:
                self.features = retrain.features
                self.classifier = retrain.classifier
            return
        if self.depth == 11:
            pretrainmodel = models.vgg11(pretrained=True)
        elif self.depth == 13:
            pretrainmodel = models.vgg13(pretrained=True)
        elif self.depth == 16:
            pretrainmodel = models.vgg16(pretrained=True)
        elif self.depth == 19:
            pretrainmodel = models.vgg19(pretrained=True)

        self.features = pretrainmodel.features
        # self.avgpool = nn.AvgPool2d(16, stride=1)
        self.classifier = nn.Sequential(torch.nn.Linear(512*16*16, 4096),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(p=0.5),
                               torch.nn.Linear(4096, 4096),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(p=0.5),
                               torch.nn.Linear(4096, self.num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.opt.trainingType == 'onevsall':
            x = self.sigmoid(x)
        return x
