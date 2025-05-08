_base_ = ["e2e_phonebackview_videomae_s_768x1_160_adapter.py"]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=16),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.9,
                crop_ratio=[0.95, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            # dict(type="mmaction.CenterCrop", crop_size=1080),
            dict(type="mmaction.Resize", scale=(-1, 240)),
            dict(type="mmaction.RandomResizedCrop", area_range=(0.95, 1.0), aspect_ratio_range=(0.95, 1.05)),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            # dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            # dict(type="mmaction.CenterCrop", crop_size=1080),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            # dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            # dict(type="mmaction.CenterCrop", crop_size=1080),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            # dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)


model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        custom=dict(pretrain="pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
    ),
    projection=dict(in_channels=768),
    rpn_head=dict(
        num_classes=12,
    ),
)

solver = dict(
    train=dict(batch_size=4, num_workers=8),
    val=dict(batch_size=2, num_workers=1),
    test=dict(batch_size=2, num_workers=1),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))

work_dir = "exps/b11_phone_motion2_backview/adatad/e2e_actionformer_videomae_b_768x1_224_adapter_s1"

workflow = dict(
    logging_interval=5,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=40,
    end_epoch=600,
)
