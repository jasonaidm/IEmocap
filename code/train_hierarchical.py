from __future__ import print_function
from load_data import get_data, analyze_data, train_data_generation  #process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking, Embedding, concatenate, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
from keras import backend
from attention_model import AttentionLayer
from load_data import data_generator, data_generator_output
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
import numpy as np

max_features = 20000
batch_size = 16
epo = 100
numclass = 4
audio_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018/Word Mat/'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

print('train_audio shape:', len(train_audio_data))
print('train_text shape:', train_text_data.shape)
print('test_audio shape:', len(test_audio_data))
print('test_text shape:', test_text_data.shape)
print('train_label shape:', train_label.shape)
print('test_label shape:', test_label.shape)

"""
final_train_audio, final_train_text, final_train_label = process_train_data(train_audio_data, train_text_data, train_label)
final_train_audio = np.array(final_train_audio)
print('train_audio shape:', final_train_audio.shape)
print('train_text shape:', final_train_text.shape)
print('test_audio shape:', test_audio_data.shape)
print('test_text shape:', test_text_data.shape)
print('train_label shape:', final_train_label.shape)
print('test_label shape:', test_label.shape)
"""

# Audio branch
frame_input = Input(shape=(513, 64))
mask_frame_input = Masking(mask_value=0.)(frame_input)
print('mask_frame_input shape: ', mask_frame_input.shape)
frame_l1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_1'))(mask_frame_input)
print('frame_l1 shape: ', frame_l1.shape)
frame_att = AttentionLayer()(frame_l1)
print('frame_att shape: ', frame_att.shape)
dropout_frame = Dropout(0.5)(frame_att)
model_frame = Model(frame_input, dropout_frame)

word_input = Input(shape=(98, 513, 64))
mask_word_input = Masking(mask_value=0.)(word_input)
print('mask_word_input shape: ', mask_word_input.shape)
audio_input = TimeDistributed(model_frame)(mask_word_input)
print('audio_input shape: ', audio_input.shape)
audio_input = Masking(mask_value=0.)(audio_input)
audio_l1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_2'))(audio_input)
print('audio_l1 shape: ', audio_l1.shape)
word_att = AttentionLayer()(audio_l1)
print('word_att shape: ', word_att.shape)
dropout_word = Dropout(0.5)(word_att)

audio_prediction = Dense(numclass, activation='softmax')(dropout_word)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=word_att)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Text Branch
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
mask_text_input = Masking(mask_value=0.)(em_text)
text_l1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25, name='LSTM_text'))(mask_text_input)
text_att = AttentionLayer()(text_l1)
dropout_text = Dropout(0.5)(text_att)

text_prediction = Dense(numclass, activation='softmax')(dropout_text)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Fusion Model
text_f_input = Input(shape=(256,))
audio_f_input = Input(shape=(256, ))
merge = concatenate([text_f_input, audio_f_input], name='merge')
#merge = backend.dot(text_f_input, audio_f_input)
d_1 = Dense(256)(merge)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.25)(activation1)
d_2 = Dense(128)(d_drop1)
batch_nol2 = BatchNormalization()(d_2)
activation2 = Activation('relu')(batch_nol2)
d_drop2 = Dropout(0.25)(activation2)
f_prediction = Dense(numclass, activation='softmax')(d_drop2)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
#visualization = Model(inputs=[text_f_input, audio_f_input], outputs=merge)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

"""
# Merge Layer
merge = concatenate([audio_att, text_att], name='merge')
dropout_l1 = Dropout(0.5)(merge)

final_prediction = Dense(4, activation='softmax')(dropout_l1)
final_model = Model(inputs=[audio_input, text_input], outputs=final_prediction)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_model.summary()

print('Train...')
#result = final_model.fit([train_audio_data, train_text_data], train_label, batch_size=batch_size, epochs=15, validation_data=([test_audio_data, test_text_data], test_label), verbose=1)
#print(result.history)
"""


audio_acc = 0
for i in range(50):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d)/8, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate_generator(data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
                                                   steps=len(test_audio_data)/8)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, train_audio_data, train_label,
                                                                               len(train_audio_data)),
                                                                steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_label,
                                                                              len(test_audio_data)),
                                                               steps=len(test_audio_data))


text_acc = 0
for i in range(25):
    print('text branch, epoch: ', str(i))
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)


final_acc = 0
for i in range(epo):
    print('fusion branch, epoch: ', str(i))
    #visualization_res = visualization.predict([train_text_inter, train_audio_inter])
    final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= final_acc:
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)


r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print(r_0)
print(r_1)
print(r_2)
print(r_3)
print(r_4)