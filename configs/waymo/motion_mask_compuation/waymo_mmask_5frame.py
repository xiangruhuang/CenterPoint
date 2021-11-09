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

#train_preprocessor = dict(
#    mode="train",
#    shuffle_points=True,
#    global_rot_noise=[0,0], #[-0.78539816, 0.78539816],
#    global_scale_noise=[1., 1.], #[0.95, 1.05],
#    db_sampler=None, #db_sampler,
#    class_names=class_names,
#    with_motion_mask=True,
#)
#
#val_preprocessor = dict(
#    mode="val",
#    shuffle_points=False,
#)

pc_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=150000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type='LoadLiDARSequence', debug=True),
    dict(type='FindMovingBoxes', debug=True),
    ##dict(type='FilterIsolatedPoints', debug=True),
    #dict(type='FilterGround', rel_threshold=0.5, debug=True, lamb=10),
    #dict(type='TemporalVoxelization', voxel_size=[0.6, 0.6, 0.6, 1], debug=True),
    #dict(type='FindConnectedComponents', radius=0.3, debug=True, granularity='points'),
    #dict(type='EstimateMotionMask', point_cloud_range=pc_range, debug=True),
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
