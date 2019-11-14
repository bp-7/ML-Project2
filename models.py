import gensim
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D


def build_model(filters, kernel_size, hidden_dims):

    model = Sequential([
        Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1),
        GlobalMaxPooling1D(),
        Dense(hidden_dims),
        Dropout(0.2),
        Activation('relu'),
        Dense(1, activation='sigmoid')]
    )

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def convert_w2v(model_gs, data, size_w2v, len_max_tweet):

    x = np.empty((data.shape[0], len_max_tweet, size_w2v))

    for idx_t, tweet in enumerate(data):
        vec = model_gs.wv[tweet]
        x[idx_t, :, :] = np.transpose(sequence.pad_sequences(vec.T, maxlen=len_max_tweet, dtype=np.float32))

    return x