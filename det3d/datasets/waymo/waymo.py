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
class WaymoDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        sequence_sampler=None,
        **kwargs,
    ):
        self.load_interval = load_interval
        if sequence_sampler is not None:
            self.sequence_sampler = sequence_sampler
        else:
            self.sequence_sampler = None
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1

    def reset(self):
        assert False 

    def select_sequences(self):
        infos = self._waymo_infos
        seq_ids = [int(info['path'].split('/')[-1].split('.')[0].split('_')[1]) for info in infos]
        selected_infos = []
        self.seq_interval = self.sequence_sampler['seq_interval']
        self.refill = self.sequence_sampler['refill']
        for info, seq_id in zip(infos, seq_ids):
            if seq_id % self.seq_interval == 0:
                selected_infos.append(info)
        print(f'Subsampled into {len(selected_infos)} frames')
        if self.refill:
            while len(selected_infos) < len(self._waymo_infos):
                for info, seq_id in zip(infos, seq_ids):
                    if seq_id % self.seq_interval == 0:
                        selected_infos.append(info)
                        if len(selected_infos) == len(self._waymo_infos):
                            break

        self._waymo_infos = selected_infos 

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        self._waymo_infos = _waymo_infos_all

        if self.sequence_sampler is not None:
            self.select_sequences()
        self._waymo_infos = self._waymo_infos[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None 

