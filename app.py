from flask import Flask
import os
from flask import send_from_directory, redirect, request, render_template, Flask, flash, url_for, session
from flask_cors import CORS
from flask_session import Session
from tempfile import gettempdir
import pyrebase
import pprint
import requests
from firebase_admin import credentials
from firebase_admin import auth
import datetime
from werkzeug.utils import secure_filename
# To preprocess uploaded essay files
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# For Google Cloud Firestore
from google.cloud import firestore
import requests
import numpy as np
import pickle
import json
from spellchecker import SpellChecker
import pprint
import string

# Web app's Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyDkoST8ntLmmPJIG00pwGp0lClJ3Z4T9rc",
    "authDomain": "essayscore.firebaseapp.com",
    "databaseURL": "https://essayscore.firebaseio.com",
    "projectId": "essayscore",
    "storageBucket": "essayscore.appspot.com",
    "messagingSenderId": "588712397910",
    "appId": "1:588712397910:web:f882bea1aa31b118a97ff7",
    "measurementId": "G-1ST0B6HD4Y"
}
# Firebase Auth
firebase = pyrebase.initialize_app(firebaseConfig)

# The minimum and maximum scored attained for each prompt
# Used in the normalisation of the score.
# First element in the list (-1) is used for padding
MIN_SCORES = [-1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
MAX_SCORES = [-1, 12, 6, 3, 3, 4, 4, 30, 60]

# Cloud Firestore
db = firestore.Client('essayscore')

app = Flask(__name__, static_url_path='/static')
# session = {}
app.secret_key = "random strings"
# ensure responses aren't cached
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

# configure session to use filesystem (instead of signed cookies)
#app.config["SESSION_FILE_DIR"] = gettempdir()
#print(app.config["SESSION_FILE_DIR"])
#app.config["SESSION_PERMANENT"] = False
#app.config["SESSION_TYPE"] = "filesystem"
#aSession(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if (request.method == "GET"):
        return render_template('index.html')
    else:
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            # Get a reference to the auth service
            auth = firebase.auth()
            user = auth.sign_in_with_email_and_password(email, password)
            pprint.pprint(user)
            # Set user session
            session['id_token'] = user['idToken']
            session['email'] = user['email']
            return redirect(url_for("dashboard"))
        except requests.exceptions.HTTPError as e:
            response = e.args[0].response
            error = response.json()['error']
            print(error)
            if error['message'] == "INVALID_PASSWORD":
                flash("Wrong Password or Username.", "danger")
            elif error['message'] == "EMAIL_NOT_FOUND":
                flash("Not a registered email.", "danger")
            else:
                flash(error['message'], "danger")
            print("Log in failed!")
            return render_template('index.html')
            
@app.route('/register', methods=["GET", "POST"])
def register():
    successful = "Registered Successfully! Please log in."
    if (request.method == "GET"):
        return redirect('index.html')
    else:
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            auth = firebase.auth()
            user = auth.create_user_with_email_and_password(email, password)
            session['id_token'] = user["idToken"]
            session['email'] = user['email']
            flash(successful, "info")
            return redirect('/')
        except requests.exceptions.HTTPError as e:
            response = e.args[0].response
            error = response.json()['error']
            if error["message"] == "EMAIL_EXISTS":
                flash("Email already exists.", "danger")
            else:
                flash(error['message'], "danger")
            print("Register failed!")
            return render_template('index.html')
        
@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    if "id_token" not in session:
        return redirect(url_for("login"))
    if "id_token" not in session:
        return redirect(url_for("login"))
    if request.method == "GET":
        flash("Login Successfully! Welcome " + session['email'] + "!", "success")
        return render_template("dashboard.html", email=session['email'])

@app.route('/logout', methods=["GET"])
def logout():
    # Get a reference to the auth service
    auth = firebase.auth()
    auth.current_user = None
    session.clear()
    flash("Successfully logged out!", "success")
    return redirect('/')

@app.route('/grading', methods=["GET"])
def grading():
    if "id_token" not in session:
        return redirect(url_for("login"))
    # retrieve the prompt
    prompt = request.args.get('prompt', None)
    session['prompt'] = prompt
    print(prompt)
    return render_template("grading.html", topic=prompt, email=session['email'])

@app.route('/upload_essay', methods=["POST"])
def upload_essay():
    if (request.method == "POST"):
        fh = request.files['file']
        if fh:
            filename = secure_filename(fh.filename)
            essay_text = [line.strip(b"\r\n") for line in fh]
            return render_template("grading.html", topic=session['prompt'], email=session['email'], essay_text=essay_text)

@app.route('/backend_grade', methods=["POST"])
def grade():
    essay_text = request.form.get("essay")
    print(essay_text)
    print("Entering backend grading")

    # Testing: Make GET requests to confirm the status of the backend
    response = requests.get("http://192.168.99.100:8501/v1/models/aes_regressor")
    print(response.json())
    # Preprocess the text to fit into the model
    # Read in the tokenizer used to convert text to sequence
    with open('tokenizer.pickle', 'rb') as handle:
        nltk_tokenizer = pickle.load(handle)
        sequence_vectors = nltk_tokenizer.texts_to_sequences([essay_text])
        print(sequence_vectors)
        # Pad the vectors so that all of them is 500 words long
        data = pad_sequences(sequence_vectors, maxlen=500)
        print(data.shape)
    # Perform spell check using pyspellchecker
    spell_check_dict = check_spellings(essay_text)
    pprint.pprint(spell_check_dict)
    # Define payload for POST request
    payload = {
        "instances": data.tolist()
    }
    # Perform POST RESTful API request to TF Serving Endpoint
    response = requests.post("http://192.168.99.100:8501/v1/models/aes_regressor:predict", json=payload)
    print(response.status_code)
    print(response.text)
    if (response.status_code == 200):
        print(int(session["prompt"]))
        print(MAX_SCORES[int(session["prompt"])])
        pred_score = response.json()["predictions"][0][0]
        # Denormalise the returned score
        true_score = denormalise_scores(pred_score, int(session["prompt"]))
        # Store the record in FireStore
        doc_ref = db.collection(u'submissions').document()
        results_dict = {
            'email': session['email'],
            'timestamp': datetime.datetime.now(),
            'text': essay_text,
            'topic': session['prompt'],
            'max_score': MAX_SCORES[int(session['prompt'])],
            'score_obtained': true_score
        }
        doc_ref.set(results_dict)
        # Display the score to UI
        return render_template("results.html", result=results_dict, email=session["email"], spelling=spell_check_dict)
    else:
        flash("Problem retrieving score from the server. Please check the console for details.", "error")
        print(response.text)
        return render_template("results.html")

def denormalise_scores(pred_score, essay_prompt):
    """ Denormalise the score returned by the model into their real score.
        Parameters:
            pred_score:     predicted score return by the model.
            essay_prompt:   the topic of the essay.
        Returns:
            Essay score scaled back to their respective topics.
    """
    true_score = pred_score * (MAX_SCORES[essay_prompt] - MIN_SCORES[essay_prompt]) + MIN_SCORES[essay_prompt]
    return true_score

@app.route('/view_submissions', methods=["GET"])
def view_submissions():
    if request.method == "GET":
        email = session['email']
        docs = db.collection('submissions').where("email", "==", email).stream()
        submissions = []
        for doc in docs:
            print('{} => {}'.format(doc.id, doc.to_dict()))
            submissions.append(doc.to_dict())
        print(submissions)
        return render_template("view_submissions.html", email=session['email'], submissions=submissions)

def check_spellings(essay):
    spell = SpellChecker(distance=1)
    contractions_str = "'tis,'twas,ain't,aren't,can't,could've,couldn't,didn't,doesn't,don't,hasn't,he'd,he'll,he's,how'd,how'll,how's,i'd,i'll,i'm,i've,isn't,it's,might've,mightn't,must've,mustn't,shan't,she'd,she'll,she's,should've,shouldn't,that'll,that's,there's,they'd,they'll,they're,they've,wasn't,we'd,we'll,we're,weren't,what'd,what's,when,when'd,when'll,when's,where'd,where'll,where's,who'd,who'll,who's,why'd,why'll,why's,won't,would've,wouldn't,you'd,you'll,you're,you've"
    spell.word_frequency.load_words(contractions_str.split(','))   
    punc_list = string.punctuation.replace('@', '')
    punc_list = punc_list.replace('\'', '')
    essay = essay.translate(str.maketrans('', '', punc_list))
    essay_words = essay.lower().split() # get each words in the text
    print(essay_words)
    spell_check_dict = {}
    for word in essay_words:
        # Skip anonymized tokens
        if word.startswith('@'):
            continue
        if word not in spell:
            spell_check_dict[word] = spell.candidates(word)
    return spell_check_dict
    
if __name__ == "__main__":
    # Set debugging to true to enable hot reload
    app.secret_key = 'any random stdring'
    app.run(debug=True)
    