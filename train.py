import torch
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

USE_BEST = False
RESUME = False

EPOCHS = 30
BATCH_SIZE = 16
WORKERS = 8

NUM_CLASSES = 1
CLASSES = ['Target']

DATASET_PARAMS = {
    'data_dir':'C:/Users/Sean/Documents/Coding/findingtgt/FindingTargets',
    'train_images_dir':'data/train/images',
    'train_labels_dir':'data/train/labels',
    'val_images_dir':'data/valid/images',
    'val_labels_dir':'data/valid/labels',
    'test_images_dir':'data/test/images',
    'test_labels_dir':'data/test/labels',
    'classes':CLASSES 
}
CHECKPOINT_DIR = "runs"

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': DATASET_PARAMS['data_dir'],
        'images_dir': DATASET_PARAMS['train_images_dir'],
        'labels_dir': DATASET_PARAMS['train_labels_dir'],
        'classes': DATASET_PARAMS['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
    )
 
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': DATASET_PARAMS['data_dir'],
            'images_dir': DATASET_PARAMS['val_images_dir'],
            'labels_dir': DATASET_PARAMS['val_labels_dir'],
            'classes': DATASET_PARAMS['classes']
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )

    # load Yolo Nas model
    MODEL_ARCH = 'yolo_nas_s'
    if USE_BEST:
        model = models.get(MODEL_ARCH, num_classes=NUM_CLASSES, checkpoint_path="ckpt_best.pth").to(DEVICE)
    else:
        model = models.get(MODEL_ARCH, num_classes=NUM_CLASSES, pretrained_weights="coco").to(DEVICE)

    # initialize the trainer for the model
    trainer = Trainer(experiment_name="target", ckpt_root_dir=CHECKPOINT_DIR)

    # define training parameters
    train_params = {
        'resume':RESUME,
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=NUM_CLASSES,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=NUM_CLASSES,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=NUM_CLASSES,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50:0.95'
    }

    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )
