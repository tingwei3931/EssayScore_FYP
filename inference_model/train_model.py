from models.LSTM_MOT import LSTM_MOT
from models.BiLSTM import BiLSTM
from preprocess import readData, tokenize
from constants import DATASET_PATH 
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
import json
from keras.models import model_from_json
from keras import backend as K
from keras.utils.vis_utils import plot_model
from time import time
import tensorflow as tf
from tensorflow.python.client import device_lib

# Configure tensorflow session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices()) # Will output list of GPUs available (CUDA Toolkit downloaded as pre-requisite)

train_set, test_set, validation_set = readData(DATASET_PATH)
# Tokenize essays into list of sentences and then into a list of wordss
# tokens = parse_essays(data['essay'])
# train word2vec model
# model = train_word2vec(tokens)

X_train, Y_train = tokenize(train_set)
X_validation, Y_validation = tokenize(validation_set)
X_test, Y_test = tokenize(test_set)

# Get LSTM Model
lstm_mot = LSTM_MOT()
model = lstm_mot.getModel()
# Get BiLSTM model
#bilstm = BiLSTM()
#model = bilstm.getModel()
# Configure tensorboard for visualisation
tensorboard = TensorBoard(log_dir='logs/LSTMMOT-{}'.format(time()), histogram_freq=2, batch_size=32, write_grads=False)
# checkpoint (only saves models that is improving the val_mse)
filepath="model_weights/LSTMMOT-weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks = [tensorboard, checkpoint]

# Train the model 
print("Begin model training...")
# Visualise model as png (not working on the new laptop)
# plot_model(model, to_file='LSTOMMOT.png', show_shapes=True, show_layer_names=True)
trained_model = model.fit(X_train, Y_train, batch_size=128, epochs=50, validation_data=(X_validation, Y_validation), callbacks=callbacks, verbose=1)
# evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
# serialize model to JSON
model_json = model.to_json()
with open("model_LSTMMOT.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("LSTMMOT_weights.h5")



