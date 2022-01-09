import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class WaymoTraceDataset(PointCloudDataset):
    NumPointFeatures = 4  # x, y, z, t

    def __init__(
        self,
        root_path,
        info_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        load_interval=1,
        num_samples=100,
        **kwargs,
    ):
        self.load_interval = load_interval
        self.sample = sample
        self._num_samples = num_samples
        super(WaymoTraceDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoTraceDataset.NumPointFeatures

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        self._waymo_infos = _waymo_infos_all
        #self._waymo_infos = [info for info in self._waymo_infos if info['cls'] != 3]

        self._waymo_infos = self._waymo_infos[::self.load_interval]
        if not self.test_mode:
            new_infos = []
            class_count = [0,0,0,0]
            for info in self._waymo_infos:
                cls = info['cls']
                if class_count[cls] < self._num_samples:
                    class_count[cls] += 1
                    new_infos.append(info)
            #max_class_count = max(class_count)
            #for i in range(4):
            #    num_duplicate = max_class_count // class_count[i]
            #    for itr in range(num_duplicate):
            #        for info in self._waymo_infos:
            #            if info['cls'] == i:
            #                new_infos.append(info)
            self._waymo_infos = new_infos

            print(f'Class Count: {class_count}')
        print("Using {} Traces".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]
        path = info['path']
        token = path.split('/')[-1].split('.')[0]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": token,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoTraceDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, labels, output_dir=None):
        TP, FP = [0, 0, 0, 0], [0, 0, 0, 0]
        for info in self._waymo_infos:
            path = info['path']
            token = path.split('/')[-1].split('.')[0]
            gt_cls = info['cls']
            pred_cls = labels[token]
            if gt_cls == pred_cls:
                TP[pred_cls] += 1
            else:
                FP[pred_cls] += 1
        for pred_cls in range(4):
            pred = TP[pred_cls] / (TP[pred_cls] + FP[pred_cls] + 1e-8)
            print(f'cls={pred_cls}, precision={pred:.4f}, TP={TP[pred_cls]}, FP={FP[pred_cls]}')

        return None, None 

