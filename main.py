from tabnanny import verbose
import spacy
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump,load
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()
    return str_text

# print(read_file('moby_dick_four_chapters.txt'))

nlp = spacy.load('en_core_web_sm', disable=['parser','tagger','ner'])
nlp.max_length = 1198623

def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

d = read_file('moby_dick_four_chapters.txt')

tokens = separate_punc(d)

train_len = 25 + 1
text_sequences = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
sequences = np.array(sequences)
vocabulary_size = len(tokenizer.word_counts) + 1

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y,num_classes=vocabulary_size)

seq_len = X.shape[1]

# model = Sequential()
# model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(vocabulary_size,activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# model.fit(X,y,batch_size=128,epochs=2,verbose=1)

def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text],maxlen=seq_len,truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

seed_text = ''

model = load_model('epochBIG.h5')
tokenizer = load(open('epochBIG','rb'))

print(generate_text(model,tokenizer,seq_len,seed_text,25))