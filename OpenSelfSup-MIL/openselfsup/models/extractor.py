import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS

@MODELS.register_module
class Extractor(nn.Module):
    def __init__(self,
                 backbone,
                 pretrained=None):
        super(Extractor, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, img, mode='extract', **kwargs):
        if mode == 'extract':
            return self.forward_extract(img)
        else:
            raise Exception("No such mode: {}".format(mode))
    
    def forward_extract(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        backbone_feats = self.avgpool(backbone_feats[-1])
        backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
        return dict(backbone=backbone_feats.cpu())