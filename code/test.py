from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional
from keras.layers import LSTM, Masking, Input, Dropout
from keras.datasets import imdb
from keras.optimizers import Adam

max_features = 20000
maxlen = 10  # cut texts after this number of words (among top max_features most common words)
a = [[[1,1],[2,2],[3,3],[4,4]],
     [[1, 1], [2, 2]]]
label = [[0,1],
         [1,0]]

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(a, maxlen=maxlen)
print(x_train)

print('x_train shape:', x_train.shape)

print('Build model...')
frame_input = Input(shape=(10, 2))
mask_frame_input = Masking(mask_value=0.)(frame_input)
frame_l1 = Bidirectional(LSTM(16, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_1'))(mask_frame_input)
frame_l2 = Bidirectional(LSTM(16, recurrent_dropout=0.25, name='LSTM_audio_2'))(frame_l1)
dropout_word = Dropout(0.5)(frame_l2)

audio_prediction = Dense(2, activation='softmax')(dropout_word)
audio_model = Model(inputs=frame_input, outputs=audio_prediction)
inter_audio = Model(inputs=frame_input, outputs=frame_l1)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.summary()


print('Train...')

for i in range(1):
    print('text branch, epoch: ', str(i))
    audio_model.fit(x_train, label, batch_size=1, epochs=1, verbose=1)
    b = inter_audio.predict(x_train, batch_size=1)
print(b)
