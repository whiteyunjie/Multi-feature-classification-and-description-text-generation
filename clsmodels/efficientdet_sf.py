import torch
from torch import nn
import math

from clsmodels.efficientmodel import BiFPN, Regressor, Classifier, EfficientNet
#from efficientdet.utils import Anchors

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

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=[2,2,2], compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        #self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                           num_layers=self.box_class_repeats[self.compound_coef],
        #                           pyramid_levels=self.pyramid_levels[self.compound_coef])
        #self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                             num_classes=num_classes,
        #                             num_layers=self.box_class_repeats[self.compound_coef],
        #                             pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.classificationModel1 = ClassificationModel(64, num_classes=num_classes[0]) #nuclei
        self.classificationModel2 = ClassificationModel(64, num_classes=num_classes[1]) #ratio
        self.classificationModel3 = ClassificationModel(64, num_classes=num_classes[2]) #growth pattern 
        #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
        #                       pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
        #                       **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)
        '''
        for name, module in model.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)

            if is_conv_layer:
                if "conv_list" or "header" in name:
                    variance_scaling_(module.weight.data)
                else:
                    nn.init.kaiming_uniform_(module.weight.data)

                if module.bias is not None:
                    if "classifier.header" in name:
                        bias_value = -np.log((1 - 0.01) / 0.01)
                        torch.nn.init.constant_(module.bias, bias_value)
                    else:
                        module.bias.data.zero_()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        #regression = self.regressor(features)
        #classification = self.classifier(features)
        #anchors = self.anchors(inputs, inputs.dtype)
        cls1 = self.classificationModel1(features[0])
        cls2 = self.classificationModel2(features[1])
        cls3 = self.classificationModel3(features[2])


        return (cls1+cls2+cls3)/3

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
