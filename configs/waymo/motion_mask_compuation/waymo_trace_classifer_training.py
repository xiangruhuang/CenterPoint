import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'OTHER']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
        type="TraceClassifier",
        backbone=dict(
            type="PointTransformer",
            in_channels=0,
            out_channels=4,
            dim_model=[32, 64, 128, 256, 512],
            k=16,
        ),
)


train_cfg = dict(
                )

test_cfg = dict(
)


# dataset settings
dataset_type = "WaymoTraceDataset"
nsweeps = 1
data_root = "data/Waymo"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/Waymo/dbinfos_subtrain_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
        dict(CYCLIST=10),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                VEHICLE=5,
                PEDESTRIAN=5,
                CYCLIST=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) 

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[0,0], #[-0.78539816, 0.78539816],
    global_scale_noise=[1., 1.], #[0.95, 1.05],
    db_sampler=None, #db_sampler,
    class_names=class_names,
    with_motion_mask=True,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=150000,
)

train_pipeline = [
    dict(type="LoadTracesFromFile", dataset=dataset_type),
    dict(type="LoadTraceAnnotations"),
    dict(type="TraceRandomRotation"),
    dict(type="ReformatTrace"),
]
test_pipeline = [
    dict(type="LoadTracesFromFile", dataset=dataset_type),
    dict(type="LoadTraceAnnotations"),
    dict(type="ReformatTrace"),
]

train_anno = "data/Waymo/infos_train_trace_classifier.pkl"
val_anno = "data/Waymo/infos_train_trace_classifier.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        class_names=class_names,
        pipeline=train_pipeline,
        load_interval=1,
        num_samples=20000,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
        load_interval=1,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="SimpleTextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 240
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
