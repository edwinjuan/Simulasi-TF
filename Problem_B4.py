# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from cgi import test
from random import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8 #done

    # YOUR CODE HERE
    # Using "shuffle=False"
    labels, sentences = bbc['category'], bbc['text']
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, shuffle=False, test_size=0.2)
    
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_train_seq = label_tokenizer.texts_to_sequences(y_train)
    label_test_seq = label_tokenizer.texts_to_sequences(y_test)
    label_train_seq_np = np.array(label_train_seq)-1
    label_test_seq_np = np.array(label_test_seq)-1

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words = vocab_size, oov_token = oov_tok)

    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(x_train)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(x_test)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction

    # Callback ygy
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_acc') is not None and logs.get('val_acc') > 0.91):
            # Stop if threshold is met
                print("\nAcc is higher than 0.9 so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model.fit(padded, label_train_seq_np, batch_size=5, epochs=50, validation_data=(testing_padded, label_test_seq_np), callbacks=[callbacks])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
