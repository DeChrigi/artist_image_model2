# Project Goal and Motivation

The aim of this project is to develop a Convolutional Neural Network (CNN) model capable of predicting the artist of a given artwork. 
This project seeks to explore the application of deep learning techniques in the field of art history and digital humanities, 
providing a tool that can assist in the classification and identification of artists based on their unique styles and techniques.

Art identification and authentication have traditionally relied on the expertise of art historians and experts who analyze the style, 
technique, and material composition of artworks. However, with the advent of digital technology and the availability of large datasets of digitized artworks, 
there is an opportunity to leverage machine learning to automate and enhance these processes.

Convolutional Neural Networks (CNNs) are particularly well-suited for image classification tasks due to their ability to capture features within images. 
By training a CNN model on a diverse dataset of artworks attributed to various artists, the model can learn to recognize patterns and features specific to each artist's style (in theory). 
This not only aids in artist identification but also provides insights into stylistic influences and the evolution of artistic techniques over time.

The development of such a model has significant implications for art conservation, curation, and academic research. 
It can serve as a valuable tool for museums, galleries, and collectors in verifying the authenticity of artworks and understanding their provenance. 
Additionally, it can support educational initiatives by offering a technological perspective on art analysis, complementing traditional methods.

# Data Collection 

The image dataset was sourced from the following URL: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time.

This dataset comprises paintings from 50 renowned artists, with each artist contributing between 24 and 877 artworks. A detailed analysis of the dataset can be found in the notebook titled "dataset_analysis."

The downloadable file, "archive.zip," contains all the paintings organized into folders corresponding to each artist. I extracted and cleansed the data to correct naming errors in the folder structure, as detailed in the script "renameData.py."

Subsequently, the paintings were divided into training and testing sets via the split_traintest.py. The training set was exclusively used to train the models, while the test set was reserved for validating the models with previously unseen paintings.

# Modeling

For the modeling phase, I employed various approaches to achieve the best possible accuracy without overfitting the data.

The model.py file is structured into different methods, which can be seen as model versioning.

Model V2: This version was my initial draft using a Convolutional Neural Network (CNN), which laid the foundation for subsequent versions.

Model V4: In this version, I experimented with adding more layers to see if it would improve accuracy. However, the it didn't impact the accuracy at all.

Model V5: I reduced the learning rate to mitigate overfitting, which resulted in a lower accuracy.

Model V6: This version utilized a pretrained VGG16 model to which I added my data. Ultimately, this model achieved the highest accuracy.

The data generation process was consistent across all models. Image augmentation positively affected validation accuracy, as without it, the models tended to overfit.

# Validation

The model_analysis notebook provides deeper insights into the models' performance. Here is a summary:

V6: Validation Accuracy: 0.45
V2: Validation Accuracy: 0.35
V4: Validation Accuracy: 0.35
V5: Validation Accuracy: 0.30

For validation, I split the data into training and validation sets before training the models. During training, all images in the ./dataset/training_images folder are used, and these are further split into training and validation sets during the epochs.

For actual validation, the images in the ./dataset/test_images folder are used. These images have never been seen by the models before, making them significant for validation.

Additionally, there is a frontend application called application.py where you can experiment with and evaluate the models' performance.

# Interpretation

While an accuracy of 45% is not sufficient for professional use, it is still significantly better than a random guess. The likely reason for the low accuracy is the imbalance in the dataset. The number of images per class ranges from 25 to 877, causing the model to be heavily influenced by certain classes. Adding more images to the underrepresented classes would likely improve the results.

Another challenge is that, although artists often follow a particular style, this is not always consistent. Artists tend to change their style over time, resulting in varied outputs. Additionally, artists create different types of works, such as portraits, landscapes, or abstract art, which further complicates the model's task when the artist has a broad repertoire.

Given the ambitious nature of the use case, I am satisfied with an accuracy of 45%.

# Tutorial for setting up the project

Since the models and the dataset are quite large, it was not possible to push everything to github, even when zipping them.

I stored the dataset and the trained models under following url: 
https://drive.google.com/drive/u/0/folders/1VAaWbAD05p-8gmAP-vrFa4cObJQJ5UzO

dataset.zip contains the dataset
trained_models contain the trained models

Just unzip the files in the root node of the project (on the same level as all the other python files are). After unzipping the files the structure should look like this:

artist_image_model2
¦
¦--dataset
¦  ¦
¦  ¦--testimages
¦  ¦--trainimages
¦
¦--trained_models
¦  ¦
¦  ¦--my_model_v2.h5
¦  ¦--my_model_v4.h5
¦  ¦--etc.
¦
¦--templates
¦  ¦
¦  ¦--index.html
¦  ¦--result.html
¦
¦--application.py
¦--dataset_analysis.ipync
etc.

after that everything should be working fine