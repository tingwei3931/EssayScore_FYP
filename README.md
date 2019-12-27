# EssayScore: Automated Essay Scoring with Deep Learning

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





## References and Acknowledgement
