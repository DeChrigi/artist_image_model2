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

Subsequently, the paintings were divided into training and testing sets. The training set was exclusively used to train the models, while the test set was reserved for validating the models with previously unseen paintings.

# Modeling

For the modelling part I took different approaches to reach the best possible accuracy without overfitting the data.

I structured the model.py in different methods which could be considered as model-versioning. 

The V2 model was my first try with a cnn, which was also the foundation for the following version.

In the V4 model i tried adding more layers to see if it would improve the accuracy, but it had a negative effect (lower accuracy)

In the V5 model i added a lower learning rate to reduce overfitting and it had a positive effect on the accuracy.

In the V6 model i took a pretrained model VG16 and added my data to it. In the end, this was the model which resulted in the highest accuracy.

The datageneration was the same for all the models, image augmentation had a positive effect on the validation accuracy since without it the model was overfitting.

# Validation

The notebook model_analysis gives deeper insights into how well the models performed. Here is TLDR version of it:

V6: Val. Accuracy: 0.45
V2: Val. Accuracy: 0.35
V4: Val. Accuracy: 0.35
V5: Val. Accuracy: 0.30

For validating on how well the models actually performed i split the data into training and validation before even training the models. When training the model it uses all pictures in the ./dataset/training_images folder and splits those into training and validation when running the epochs. 

For the actual validation of the models the pictures in the folder ./dataset/test_images are used. The models have never seen those pictures before and are therefore significant for validation.

There is also a frontend application called 'application.py' in which you can experiment areound on how well the models perform.

# Interpretation

While an accuracy of 45% is not high enough to use the model in a professional environment, it is still better than an uneducated guess by a lot. The cause for the low accuracy probably lies in the fact that the dataset is not well balanced. There is a spread from 25 to 877 pictures depending on the class, which results in a model which is heavily influenced by certain classes. When adding more pictures to the undersupplied classes the result would likely be better.

Another challenge is while an artist follows a style, this is not always the case. Artists tend to change their style over time which results in different results. Artists also tend to drawdifferent motives like portraits, landscapes or abstract art. So the model has a difficult time if the artist has a broad repertoire.

The usecase itself was quite ambitious so im happy with an accuracy of 45%.
