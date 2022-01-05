from ..registry import CLASSIFIERS
from .simple import SimpleClassifier 
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
import torch.nn.functional as F

@CLASSIFIERS.register_module
class TraceClassifier(SimpleClassifier):
    def __init__(
        self,
        backbone,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(TraceClassifier, self).__init__(
            backbone, train_cfg, test_cfg, pretrained
        )
        self.vis = Visualizer([], [])

    def forward(self, example, return_loss=True, **kwargs):
        points = example['points']
        preds = self.backbone(points)

        if return_loss:
            return self.loss(example, preds)
        else:
            return self.predict(example, preds, self.test_cfg)

    def loss(self, example, preds):
        gt_labels = example['classes'].to(preds.device)
        loss = F.nll_loss(preds, gt_labels,
                          weight=torch.tensor([1.0,1.0,10.0,1.0]).to(preds.device))
        loss_dict = dict()
        loss_dict['loss'] = [loss, loss, loss, loss]
        labels = preds.argmax(-1)
        
        TP = [torch.tensor(0.0).to(preds.device) for i in range(4)]
        FP = [torch.tensor(0.0).to(preds.device) for i in range(4)]
        FN = [torch.tensor(0.0).to(preds.device) for i in range(4)]
        T = [torch.tensor(0.0).to(preds.device) for i in range(4)]

        for i in range(4):
            tp_mask = (labels == i) & (gt_labels == i)
            fp_mask = (labels == i) & (gt_labels != i)
            fn_mask = (labels != i) & (gt_labels == i)
            TP[i] += tp_mask.float().sum()
            FP[i] += fp_mask.float().sum()
            FN[i] += fn_mask.float().sum()
            T[i] += (gt_labels == i).sum()
        
        loss_dict['TP'] = TP
        loss_dict['FP'] = FP
        loss_dict['FN'] = FN
        loss_dict['T'] = T

        return loss_dict

    def predict(self, example, preds):
        pass

