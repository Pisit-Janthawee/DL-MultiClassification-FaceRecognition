import tensorflow as tf
import keras
import numpy as np
import random
import cv2
import os


def recognize(image):
    # Processing
    resized_image = cv2.resize(image, (128, 128)) / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)
    
    class_labels = ['Dong', 'Aum', 'TJ', 'Mark']
    model = tf.keras.models.load_model(
        r'D:\Repo\DS\Classification\Multi-Classification\face_recognition\artifacts\cnn_model.h5')
    probability = model.predict(resized_image).flatten()
    output = {class_labels[i]: float(probability[i]) for i in range(4)}
    return output 


def sampling_example_image():
    image_dir = r"D:\Repo\DS\Classification\Multi-Classification\face_recognition\Images"
    image_paths = [os.path.join(image_dir, f"{i}.png") for i in range(4)]
    random.shuffle(image_paths)
    return image_paths
