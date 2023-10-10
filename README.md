# DL-MultiClassification-FaceRecognition

# **Face Recognition Using Convolutional Neural Networks (CNNs)**
<img src="https://github.com/Pisit-Janthawee/DL-MultiClassification-FaceRecognition/blob/main/gradio_interface.png" align="center">

## **Introduction**

Face recognition is a crucial application in computer vision with a wide range of practical uses. In this notebook, we will explore the implementation of a Convolutional Neural Network (CNN) for multi-class face recognition using TensorFlow. The dataset used contains images of 4 individuals, labeled as Target 1, 2, 3, and 4.

## **Machine Learning Steps**

Before diving into the specifics of our face recognition project, let's outline the general machine learning steps we will follow:

1. **Data Collection**: Gather the dataset of face images.
2. **Data Preprocessing**: Prepare the data for training by resizing images and converting them to numerical format.
3. **Data Augmentation**: Enhance the dataset by applying data augmentation techniques.
4. **Model Building**: Construct a CNN architecture for face recognition.
5. **Model Training**: Train the CNN on the prepared dataset.
6. **Model Evaluation**: Evaluate the model's performance using appropriate metrics.
7. **Cross-Validation**: Perform cross-validation to ensure the model's generalization.
8. **Visualization**: Visualize the results, including comparing actual vs. model predictions.

Now, let's dive into each of these steps in detail.

## **Objective**

The primary objective of this project is to perform multi-class classification for face recognition. Specifically, we aim to:

- Develop a deep learning model based on Convolutional Neural Networks (CNNs).
- Train the model to recognize the faces of four individuals labeled as Target 1, 2, 3, and 4.
- Evaluate the model's accuracy and generalization capabilities.

## **Problem**

The problem at hand is a multi-class classification task. Given an input image, our model must correctly classify it into one of the four classes, representing the four individuals.
## **Project's Purpose and Scope**

### **Purpose**


**The primary purpose** of this project is aims to demonstrate the capabilities of deep learning in recognizing and classifying faces of four individuals.

**Demonstration**: Showcase the potential of neural networks in solving real-world problems, such as facial recognition.

### **Scope**

**Data Transformation**: Preprocessing steps involve resizing the images to 128x128 pixels and converting them into a suitable format for neural network input. (Resource constraints)


# File Description

## Folder

1. **artifacts**
   - _Explanation_: This folder may contain saved model weights, configurations, or any artifacts resulting from training and experimentation. It's where you store the final trained model for use in deployment.
2. **images**
   - _Explanation_: This folder can store image files used for explanatory purposes in Jupyter notebooks or any other documentation.

## 01-02 .ipynb Files

1. **01_init_notebook.ipynb**
   - _Explanation_: This initial notebook is used for exploring the data and performing data preprocessing tasks as outlined in the "Data Preprocessing" section. Also, CNN model is constructed in this notebook 
2. **02_deployment.ipynb**
   - _Explanation_: This notebook focuses on model deployment using Gradio. It provides an easy-to-use interface for displaying input images and model predictions as probabilities.

