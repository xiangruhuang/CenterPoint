import torch.nn as nn
from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class SNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def forward(self, example, return_loss=True, **kwargs):
        import ipdb; ipdb.set_trace()
        pass
    
    def predict(self, example, preds_dicts):
        import ipdb; ipdb.set_trace()
        pass
    
