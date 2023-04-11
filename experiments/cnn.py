"""Example code that uses a convolutional neural network to recognize cards."""

# Required dependencies:
# python = ">=3.10, <3.12"
# efficientnet = "^1.1"
# tensorflow = {version = "<=2.12", markers = "sys_platform != 'darwin'"}
# tensorflow-macos = {version = "<=2.12", markers = "sys_platform == 'darwin'"}
# tensorflow-metal = {version = "<=0.8", markers = "sys_platform == 'darwin'"}

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
from PIL import Image
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGES_DIR = "images"
BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = (936, 672)
NUM_CLASSES = 33844
MODEL_PATH = "magic_card_recognition_model.h5"

# Note that to run this code you need to have downloaded images from Scryfall to
# the IMAGES_DIR directory. Each file must be located in a subdirectory named
# after the card's scryfall id, containing an single image named image.jpg.
# The scryfall ids become the class labels for the model.


def train_model():
    # Image Augmentation (to generate varied training data from single images)
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda img: cv.blur(img, (5, 5)),
        channel_shift_range=150.0,
        rescale=1.0 / 255,
        rotation_range=90,
        shear_range=5.0,
        validation_split=0.2,
    )
    train_generator = train_datagen.flow_from_directory(
        IMAGES_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    # TODO: Save label map to file?
    # label_map = (train_generator.class_indices)
    validation_generator = train_datagen.flow_from_directory(
        IMAGES_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )
    base_model = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    predictions = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
    )
    model.save(MODEL_PATH)
    print("Training complete!")


def preprocess_image(path: str):
    img = Image.open(path)
    img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def recognize_magic_card(path: str):
    paths = sorted(f"{f}/image1.jpg" for f in Path(IMAGES_DIR).glob("*") if f.is_dir())
    paths_by_index = dict(zip(range(len(paths)), paths))
    model = tf.keras.models.load_model(MODEL_PATH)
    img = preprocess_image(path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return paths_by_index[predicted_class]


if __name__ == "__main__":
    train_model()
    print(recognize_magic_card("test_image.jpg"))
