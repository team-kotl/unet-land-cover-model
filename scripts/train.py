import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm

# Parameters
DATA_DIR = "data"
MODEL_DIR = "models"
PATCH_SIZE = (256, 256)
NUM_CLASSES = 10  # Adjust based on your classes
BATCH_SIZE = 8
EPOCHS = 50

# Data generators
data_gen_args = dict(rescale=1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

train_image_gen = image_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train/images"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
train_mask_gen = mask_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train/masks"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
train_gen = zip(train_image_gen, train_mask_gen)

val_image_gen = image_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val/images"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
val_mask_gen = mask_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val/masks"),
    target_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42
)
val_gen = zip(val_image_gen, val_mask_gen)

# Define U-Net model
model = sm.Unet('resnet34', input_shape=(*PATCH_SIZE, 3), classes=NUM_CLASSES, activation='softmax')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', sm.metrics.IOUScore()])

# Train the model
model.fit(
    train_gen,
    steps_per_epoch=len(train_image_gen),
    validation_data=val_gen,
    validation_steps=len(val_image_gen),
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "unet_best.h5"),
            save_best_only=True,
            monitor='val_iou_score',
            mode='max'
        )
    ]
)

print("Training completed.")