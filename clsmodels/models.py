import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clsmodels.net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from clsmodels.net.senet import se_resnext50_32x4d, se_resnet50
from clsmodels.net.densenet import densenet169,densenet121
#import settings
import  pretrainedmodels

class InclusiveNet(nn.Module):
    def __init__(self, backbone_name, num_classes=7272, pretrained=True):
        super(InclusiveNet, self).__init__()
        if backbone_name == 'se_resnext50_32x4d':
            self.backbone = se_resnext50_32x4d()
        elif backbone_name == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise ValueError('unsupported backbone name {}'.format(backbone_name))
        self.backbone.last_linear = nn.Linear(2048, num_classes) # for model convert

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(2048, num_classes)
        self.logit_num = nn.Linear(2048, 1)
        self.name = 'InclusiveNet_' + backbone_name
    
    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, p=0.4)
        x = x.view(x.size(0), -1)
        return self.logit(x), self.logit_num(x)
    
    def forward(self, x):
        x = self.backbone.features(x)
        return self.logits(x)


def create_backbone_model(pretrained, num_classes=1):
    if pretrained:
        #basenet = se_resnext50_32x4d()
        basenet = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        #basenet = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained=0)
        new_last_linear = nn.Linear(basenet.last_linear.in_features, num_classes)
        basenet.last_linear = new_last_linear
        basenet.avg_pool = nn.AdaptiveAvgPool2d(1)
    else:
        basenet = se_resnext50_32x4d(num_classes=num_classes, pretrained=None)

    basenet.num_ftrs = 2048
    basenet.name = 'se_resnext50_32x4d_pnasnet5large'
    return basenet

def create_model(backbone_name, pretrained, num_classes, load_backbone_weights=False):
    if backbone_name == 'se_resnext50_32x4d':
        backbone = create_backbone_model(1,num_classes=num_classes)
        #backbone = resnet10()
        
    elif backbone_name == 'resnet34':
        backbone, _ = create_pretrained_resnet(34, num_classes)
    elif backbone_name == 'resnet50':
        backbone, _ = create_pretrained_resnet(50, num_classes)
    elif backbone_name == 'resnet18':
        backbone, _ = create_pretrained_resnet(18, num_classes)
    elif backbone_name == 'densenet169':
        backbone = create_pretrained_densenet(169,num_classes)
    elif backbone_name == 'densenet121':
        backbone = create_pretrained_densenet(121,num_classes)
    else:
        raise ValueError('unsupported backbone name {}'.format(backbone_name))
    #backbone.name = backbone_name +"_in6"
    #backbone.name = backbone_name +"_he_yuan_noseg_trainandtrain2"
    backbone.name = backbone_name +"_hcc"

    #if pretrained:
    #    model_file = os.path.join(settings.MODEL_DIR, 'backbone', backbone.name, 'pretrained', 'best.pth')
    #else:
    #    model_file = os.path.join(settings.MODEL_DIR, 'backbone', backbone.name, 'scratch', 'best.pth')
    
    #if load_backbone_weights:
    #    print('loading {}...'.format(model_file))
    #    backbone.load_state_dict(torch.load(model_file))
    
    #if backbone_name == 'se_resnext50_32x4d':
    #    backbone.last_linear = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(backbone.num_ftrs, num_classes))
    #elif backbone_name == 'resnet34':
    #    backbone.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(backbone.num_ftrs, num_classes))
    
    return backbone

def create_resnet_model(layers, num_classes):
    if layers not in [18, 32, 34, 50, 101, 152]:
        raise ValueError('Wrong resnet layers')

    return eval('net.resnet.resnet'+str(layers))(pretrained=False, num_classes=num_classes)
    
def create_pretrained_resnet(layers, num_classes):
    print('create_pretrained_resnet', layers)
    if layers == 34:
        model, bottom_channels = resnet34(pretrained=True), 512
    elif layers == 18:
        model, bottom_channels = resnet18(pretrained=True), 512
    elif layers == 50:
        model, bottom_channels = resnet50(pretrained=True), 2048
    elif layers == 101:
        model, bottom_channels = resnet101(pretrained=True), 2048
    elif layers == 152:
        model, bottom_channels = resnet152(pretrained=True), 2048
    else:
        raise NotImplementedError('only 34, 50, 101, 152 version of Resnet are implemented')

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.num_ftrs, num_classes)) #这个dropout意义不大感觉
    #model.fc = nn.Linear(model.num_ftrs, num_classes)

    return model, bottom_channels


def create_pretrained_densenet(layers,num_classes):
    print('create_pretrained_densenet', layers)
    if layers == 169:
        model = densenet169(pretrained=True)
    if layers == 121:
        model = densenet121(pretrained=True)

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_features = model.classifier.in_features
    #print(num_features)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, num_classes)) #这个dropout意义不大感觉
    #model.classifier = nn.Linear(model.num_features, num_classes)

    return model


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.size())
        return x

class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2

class AttentionResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes=100, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(AttentionResNet, self).__init__()
        self.name = 'AttentionResNet_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_pretrained_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout2d(p=self.dropout_2d),
            nn.Linear(bottom_channel_nr, 100)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        x = self.encoder2(x) #; print('e2:', e2.size())
        x = self.encoder3(x) #; print('e3:', e3.size())
        x = self.encoder4(x) #; print('e4:', e4.size())
        x = self.encoder5(x) #; print('e5:', x.size())
        x = F.dropout2d(x, p=self.dropout_2d)
        x = self.avgpool(x) #; print('out:', x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    model = AttentionResNet(50).cuda()
    model.freeze_bn()
    inputs = torch.randn(2,3,128,128).cuda()
    out, _ = model(inputs)
    #print(model)
    print(out.size()) #, cls_taret.size())
    #print(out)

def test2():
    #model = create_model('se_resnext50_32x4d', pretrained=False, num_classes=100, load_backbone_weights=False).cuda()
    model = create_model('resnet34', pretrained=False, num_classes=100, load_backbone_weights=False).cuda()
    x = torch.randn(2,3,256,256).cuda()
    y = model(x)
    print(y.size())

def test3():
    model = InclusiveNet('se_resnext50_32x4d').cuda()
    x = torch.randn(2,3,256,256).cuda()
    y1, y2 = model(x)
    print(y1.size(), y2.size())

def convert_model():
    #model_file = r'G:\inclusive\models\backbone\se_resnext50_32x4d\pretrained\best.pth'
    model_file = os.path.join(settings.MODEL_DIR, 'backbone', 'se_resnext50_32x4d', 'pretrained', 'best.pth')
    old_model = create_backbone_model(True)
    old_model.load_state_dict(torch.load(model_file))

    new_model = InclusiveNet('se_resnext50_32x4d')
    new_model.backbone = old_model
    new_model.logit = old_model.last_linear
    print(new_model.backbone.last_linear)


    new_model_file = os.path.join(settings.MODEL_DIR, 'backbone', 'se_resnext50_32x4d', 'pretrained', 'best_new.pth')
    torch.save(new_model.state_dict(), new_model_file)

    new_model = new_model.cuda()
    x = torch.randn(2,3,256,256).cuda()
    y1, y2 = new_model(x)
    print(y1.size(), y2.size())


if __name__ == '__main__':
    #test()
    #test3()
    convert_model()
