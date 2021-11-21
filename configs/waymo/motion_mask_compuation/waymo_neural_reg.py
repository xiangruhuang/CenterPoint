import itertools
import logging

# dataset settings
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "data/Waymo"

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

flownet=dict(type='TFlowNet',
             channels = [(4, 128), (128, 128), (128, 128), (128, 128),
                         (128, 128), (128, 128), (128, 3)])

lr_config=dict(lr_max=1e-3,
               total_step=10000,
               moms=[0.95, 0.85],
               div_factor=10.0, pct_start=0.4)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type='LoadLiDARSequence', debug=False, load_temp=False),
    dict(type='FilterGround', rel_threshold=0.5, debug=False, lamb=10),
    #dict(type='TemporalVoxelization', voxel_size=[0.6, 0.6, 0.6, 1],
    #     debug=False),
    #dict(type='FindConnectedComponents', radius=0.3, debug=False, granularity='points'),
    #dict(type='Registration', radius=0.3, debug=True),
    dict(type='NeuralRegistration',
         flownet=flownet,
         hash_table_size=500000,
         voxel_size=[2.0, 2.0, 2.0, 1],
         lr_config=lr_config,
         resume=False,
         debug=False),
    #dict(type='FindMovingBoxes', debug=False),
    ##dict(type='FilterIsolatedPoints', debug=True),
]

train_anno = "data/Waymo/infos_train_sequences_filter_zero_gt.pkl"
val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(),
    test=dict(),
)

log_level = "INFO"
