# EssayScore: Automated Essay Scoring with Deep Learning (Final Year Project)

EssayScore is a web application that is designed to alleviate the burden of teachers in marking essays and to provide a platform for students to gain instant feedback for their essays. Implemented using Keras with TensorFlow backend for the model training and Flask microframework for the web application.  

## Getting Started

Codes for building the model are in `inference_model` folder. 
Install pip dependencies from the requirements.txt.
```
pip install -r requirements.txt
```
Run `python train_model.py` to train the model. 

Codes for the web application are in the root.
To start Flask server, run
```
python app.py
```
in the root folder. 

## Features

* Essays submitted will be graded through the neural network model for evaluation purposes.
* View past essay submissions to improve essay writing skills. 
* Checks the essay submitted for spelling errors and recommends correction to the mispelled word.

## Built With

* [Visual Studio Code](https://developer.android.com/studio) - The IDE used during development.
* [Firebase](https://firebase.google.com/) - NoSQL Cloud database used to store user data.
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - Used together with Docker container to serve 
  the trained model as a RESTful Web API.
* [Keras Framework](https://keras.io) - For training the deep learning model with TensorFlow as backend.
* [Flask Microframework](http://flask.palletsprojects.com/en/1.1.x/) - A WSGI web micro framework for developing web applications.

## Sample Screenshots

![image](https://user-images.githubusercontent.com/16443687/71521684-b4e50900-28fc-11ea-832f-9022b75975f5.png)
![image](https://user-images.githubusercontent.com/16443687/71522253-d941e500-28fe-11ea-9313-439b42ce6a59.png)
![image](https://user-images.githubusercontent.com/16443687/71522280-f5de1d00-28fe-11ea-9ab5-93a01d483029.png)
![image](https://user-images.githubusercontent.com/16443687/71522342-481f3e00-28ff-11ea-8aa1-e038d47be010.png)
![image](https://user-images.githubusercontent.com/16443687/71522403-8a487f80-28ff-11ea-96fc-a0714ce3ed8b.png)


## References 

1. [A Neural Approach to Automated Essay Scoring](https://www.aclweb.org/anthology/D16-1193.pdf) (Taghipour and Ng, 2016)
2. [Automated Text Scoring Using Neural Networks](https://arxiv.org/pdf/1606.04289.pdf) (Alikaniotis, Yannakoudakis and Rei, 2016)
3. [Robust Trait-Specific Essay Scoring Using Neural Networks and Density Estimators](https://scholarbank.nus.edu.sg/handle/10635/138207) (Taghipour, 2017)

## Acknowledgement 
1. Dr. J. Joshua Thomas for supervision and guidance throughout the project.
