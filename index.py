import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# Function to generate Siamese pairs
def generate_siamese_pairs(positive_images, negative_images, image_size):
    # Generate positive pairs (pairs of your face)
    pairs = []
    labels = []
    for img1 in positive_images:
        for img2 in positive_images:
            pairs.append([load_and_preprocess_image(img1, image_size), load_and_preprocess_image(img2, image_size)])
            labels.append(1)  # 1 indicates a positive pair

    # Generate negative pairs (pairs of your face and different people's faces)
    for img1 in positive_images:
        for img2 in negative_images:
            pairs.append([load_and_preprocess_image(img1, image_size), load_and_preprocess_image(img2, image_size)])
            labels.append(0)  # 0 indicates a negative pair

    return np.array(pairs), np.array(labels)

# Load your face images
your_face_folder = "C:\\Users\\admin\\Desktop\\DS&ML\\FaceRecog\\kevin_faces"
your_face_images = [os.path.join(your_face_folder, filename) for filename in os.listdir(your_face_folder)]

# Load different people's face images
different_people_folder = "C:\\Users\\admin\\Desktop\\DS&ML\\FaceRecog\\data"
different_people_images = [os.path.join(different_people_folder, filename) for filename in os.listdir(different_people_folder)]

# Split the data into training and validation sets
your_face_train, your_face_val = train_test_split(your_face_images, test_size=0.2)
different_people_train, different_people_val = train_test_split(different_people_images, test_size=0.2)

# Generate Siamese pairs for training and validation
image_size = (160, 160)
train_pairs, train_labels = generate_siamese_pairs(your_face_train, different_people_train, image_size)
val_pairs, val_labels = generate_siamese_pairs(your_face_val, different_people_val, image_size)

# Define the Siamese model (same as in Step 3)
siamese_model = create_siamese_model(image_size)

# Define a custom loss function (contrastive loss)
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Compile the model with the custom loss function
siamese_model.compile(optimizer="adam", loss=contrastive_loss)

# Train the Siamese model
siamese_model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]],
    train_labels,
    batch_size=32,
    epochs=10,
    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels)
)
