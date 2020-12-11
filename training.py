# Voir >>> https://www.tensorflow.org/tutorials/text/text_generation
# TensorFlow IMPLEMENTATION
import tensorflow as tf
# Je sais pas pourquoi mais avec les lignes suivantes ça fait marcher le code avec CUDA et cuDNN
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Import model and utils
import os
from model import model
from preprocessing import dataset

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# training procedure config with "Adam" optimizer
model.compile(optimizer='adam', loss=loss)
model.summary()

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# 10 époques d'entraînements pour que le temps de formation soit raisonnable
EPOCHS = 200
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
