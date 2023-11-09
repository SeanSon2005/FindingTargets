from super_gradients.training import models

MODEL_ARCH = 'yolo_nas_s'
NUM_CLASSES = 1
CHECKPOINT_PATH = "ckpt_best.pth"
DEVICE = 'cuda'

model = models.get(MODEL_ARCH, num_classes=NUM_CLASSES, checkpoint_path=CHECKPOINT_PATH).to(DEVICE)

model.predict_webcam()
#prediction = model.predict('data/train/images/Image2.jpg', fuse_model=False)
#prediction.show()

