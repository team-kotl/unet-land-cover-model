import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
import matplotlib.pyplot as plt

# Parameters
DATA_DIR = "data"
MODEL_DIR = "models"
PATCH_SIZE = (256, 256)
NUM_CLASSES = 10
BATCH_SIZE = 8

# Load test data
data_gen_args = dict(rescale=1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

test_image_gen = image_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test/images"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
test_mask_gen = mask_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test/masks"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
test_gen = zip(test_image_gen, test_mask_gen)

# Load trained model
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "unet_best.h5"),
    custom_objects={'iou_score': sm.metrics.IOUScore()}
)

# Evaluate
results = model.evaluate(test_gen, steps=len(test_image_gen))
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}, Test IoU: {results[2]}")

# Visualize predictions (optional)
images, masks = next(test_gen)
preds = model.predict(images)
for i in range(min(3, BATCH_SIZE)):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(images[i])
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(np.argmax(masks[i], axis=-1))
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(np.argmax(preds[i], axis=-1))
    plt.savefig(f"prediction_{i}.png")