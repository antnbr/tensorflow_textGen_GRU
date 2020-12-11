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
import datetime
from preprocessing import char2idx, idx2char
from model import build_model, vocab_size, embedding_dim, rnn_units
from training import checkpoint_dir

# Pour simplifier cette étape de prédiction, utilisez une taille de lot de 1.
# En raison de la façon dont l'état RNN est passé d'un pas de temps à un autre, le modèle n'accepte qu'une taille de lot fixe une fois construit.
# Pour exécuter le modèle avec un batch_size différent, vous devez reconstruire le modèle et restaurer les poids à partir du point de contrôle.
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# This block generate text according its food
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 10000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.7

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

output = generate_text(model, start_string=u"Lundi 10 Janvier,")

date = str(datetime.datetime.now())
ext = '.txt'
file_name = './generated_texts/gen_' + date + ext
text_file = open(file_name, 'w')
text_file.write(output)
text_file.close()
