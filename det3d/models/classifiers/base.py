import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn
from det3d import torchie

class BaseClassifier(nn.Module):
    """Base class for classifiers"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.fp16_enabled = False

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))

    def forward_test(self, imgs, **kwargs):
        pass

    def forward(self, example, return_loss=True, **kwargs):
        pass
