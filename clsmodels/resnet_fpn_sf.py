### code was modified from retinanet
### from https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet



import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

from clsmodels.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
#from retinanet.anchors import Anchors
#from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # these two layers may not be necessary
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=4, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        #self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()

        #self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.act3 = nn.ReLU()
        
        ## 简化成三层+fc
        #self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.act4 = nn.ReLU()

        # Average pooler
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        # Linear classifier. 
        self.fc = nn.Linear(feature_size, num_classes)
        #self.output_act = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.bn1(x)
        out = self.act1(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.act2(out)

        out = self.avg_pool(out).view(x.shape[0],-1)
        out = self.dropout(out)
        out = self.fc(out)
        #out = self.output_act(out)

        '''
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        '''
        return out

class Resnet_fpn_classifier(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(Resnet_fpn_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #(b,64,w/2,h/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.classificationModel1 = ClassificationModel(256, num_classes=num_classes[0]) #nuclei
        self.classificationModel2 = ClassificationModel(256, num_classes=num_classes[1]) #ratio
        self.classificationModel3 = ClassificationModel(256, num_classes=num_classes[2]) #growth pattern 

        # loss 函数重写, 参考miccai那个
        #self.Loss = losses.FocalLoss()

        # weights initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        #self.classificationModel1.output.weight.data.fill_(0)
        #self.classificationModel1.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        #self.classificationModel2.output.weight.data.fill_(0)
        #self.classificationModel2.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        #self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None #resnet50及以上block.expansion=4,前面都是1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        # simplify training process,BN has little influence
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        #print(inputs.size())
        img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#(b,64,w/4,h/4)

        x1 = self.layer1(x)#(b,64,w/4,h/4)
        x2 = self.layer2(x1)#(b,128,w/8,h/8)
        x3 = self.layer3(x2)#(b,256,w/16,h/16)
        x4 = self.layer4(x3)#(b,512,w/32,h/32)

        features = self.fpn([x2, x3, x4]) #[128,256,512] /[512,1024,2048]resnet>=50

        cls1 = self.classificationModel1(features[0])
        cls2 = self.classificationModel2(features[1])
        cls3 = self.classificationModel3(features[2])

        # 从3个尺度取平均，但是考虑到不同特征，可能要加不同的权重
        #return (cls1+cls2+cls3)/3
        return (cls1+cls2+cls3)/3
        #regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        #classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        #anchors = self.anchors(img_batch)

        # 后期考虑
class Resnet_fc_classifier(nn.Module):
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(Resnet_fc_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #(b,64,w/2,h/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        #self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.classificationModel1 = ClassificationModel(512, num_classes=num_classes[0]) #nuclei
        self.classificationModel2 = ClassificationModel(512, num_classes=num_classes[1]) #ratio
        self.classificationModel3 = ClassificationModel(512, num_classes=num_classes[2]) #growth pattern 

        # loss 函数重写, 参考miccai那个
        #self.Loss = losses.FocalLoss()

        # weights initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        #self.classificationModel1.output.weight.data.fill_(0)
        #self.classificationModel1.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        #self.classificationModel2.output.weight.data.fill_(0)
        #self.classificationModel2.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        #self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None #resnet50及以上block.expansion=4,前面都是1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        # simplify training process,BN has little influence
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        #print(inputs.size())
        img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#(b,64,w/4,h/4)

        x1 = self.layer1(x)#(b,64,w/4,h/4)
        x2 = self.layer2(x1)#(b,128,w/8,h/8)
        x3 = self.layer3(x2)#(b,256,w/16,h/16)
        x4 = self.layer4(x3)#(b,512,w/32,h/32)

        #features = self.fpn([x2, x3, x4]) #[128,256,512] /[512,1024,2048]resnet>=50

        cls1 = self.classificationModel1(x4)
        cls2 = self.classificationModel2(x4)
        cls3 = self.classificationModel3(x4)
        
        return cls1, cls2, cls3 




def resnet18_fpn_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fpn_classifier(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def resnet34_fc_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fc_classifier(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet34_fpn_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fpn_classifier(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50_fpn_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fpn_classifier(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101_fpn_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fpn_classifier(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152_fpn_classifier(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet_fpn_classifier(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model