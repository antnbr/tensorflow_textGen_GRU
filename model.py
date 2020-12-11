# Voir >>> https://www.tensorflow.org/tutorials/text/text_generation
# TensorFlow IMPLEMENTATION
import tensorflow as tf
# Je sais pas pourquoi mais avec les lignes suivantes ça fait marcher le code avec CUDA et cuDNN
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Import NumPy for shaping datasets and other utils
import numpy as np
import os
import time
from preprocessing import dataset, vocab, BATCH_SIZE

# ---- BUILDING THE MODEL
# Length of the vocabulary
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024

#  Utilisez tf.keras.Sequential pour définir le modèle. Pour cet exemple simple, trois couches sont utilisées pour définir notre modèle:
#  - tf.keras.layers.Embedding : La couche d'entrée. Une table de recherche entraînable qui mappera les nombres de chaque caractère à un vecteur avec des dimensions embedding_dim ;
#  - tf.keras.layers.GRU : Un type de RNN avec des units=rnn_units taille units=rnn_units (Vous pouvez également utiliser une couche LSTM ici.)
#  - tf.keras.layers.Dense : La couche de sortie, avec les sorties vocab_size
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,
								  embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
