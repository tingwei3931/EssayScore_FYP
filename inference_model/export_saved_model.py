import tensorflow as tf
from keras.models import model_from_json
from keras import backend as K 
from models.LSTM_MOT import LSTM_MOT
from tensorflow.python.platform import gfile

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
export_path = 'aes_regressor/1'
# load the model weights
model = tf.keras.models.load_model('model_LSTMMOT.h5')
print(model.outputs)
print(model.inputs)
# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

