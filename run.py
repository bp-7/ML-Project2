from helpers import *
import gensim
import keras
from keras.models import load_model
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


text_data, labels, text_data_test = get_raw_data('', 'f')

path_w2v = 'w2v_models/'
name_w2v = 'w2v_model_best'
word_vector = gensim.models.KeyedVectors.load(path_w2v + name_w2v)

# Convert gensim word_vector in keras embedding
# Choose or not to continue embedding training during network training
train_emb = True
k_emb = word_vector.get_keras_embedding(train_embeddings=train_emb)
size_emb = k_emb.output_dim

# Convert text to numerical data according to gensim (now keras embedding) vocabulary
vocabulary = {word: vector.index for word, vector in word_vector.vocab.items()}
tk = Tokenizer(num_words=len(vocabulary))
tk.word_index = vocabulary
num_data = np.asarray((pad_sequences(tk.texts_to_sequences(text_data), padding='post')))
num_data_test = np.asarray((pad_sequences(tk.texts_to_sequences(text_data_test),
                                          maxlen=num_data.shape[1], padding='post')))


model = load_model('best_model.hdf5')

# Predict on test set
y_pred = np.ndarray.flatten(model.predict_classes(num_data_test, batch_size=150))

# Replace for submission
y_pred = np.where(y_pred == 0, -1, y_pred)

# Generate submission
csv_name ='sub_best'

create_csv_submission(y_pred, csv_name + '.csv')
print("Output name:", csv_name + '.csv')
