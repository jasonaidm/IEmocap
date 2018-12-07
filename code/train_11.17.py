from __future__ import print_function

from self_attention import Attention, Position_Embedding
from load_ori_data import get_data, analyze_data  # , train_data_generation  #process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking, Embedding, concatenate, \
    GlobalAveragePooling1D, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
from attention_model import AttentionLayer
import numpy as np

max_features = 20000
batch_size = 16
epo = 100

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

print('train_audio shape:', train_audio_data.shape)
print('train_text shape:', train_text_data.shape)
print('test_audio shape:', test_audio_data.shape)
print('test_text shape:', test_text_data.shape)
print('train_label shape:', train_label.shape)
print('test_label shape:', test_label.shape)

# Audio branch
''''
word_input = Input(shape=(2250,64))
word_att = Attention(4, 16)([word_input, word_input, word_input])
word_att = BatchNormalization()(word_att)

word_att1 = Attention(4, 16)([word_att, word_att, word_att])
word_att1 = BatchNormalization()(word_att1)

#word_att2 = Attention(4, 16)([word_att1, word_att1, word_att1])
#word_att2 = BatchNormalization()(word_att2)
#word_att2 = GlobalAveragePooling1D()(word_att2)

dropout_audio = Dropout(0.5)(word_att1)
model_frame = Model(word_input, dropout_audio)
'''
audio_input = Input(shape=(2250, 64))
#word_input = TimeDistributed(model_frame)(word_input)############

audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att = BatchNormalization()(audio_att)

audio_att1 = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att1 = BatchNormalization()(audio_att1)

audio_att2 = Attention(4, 16)([audio_att1, audio_att1, audio_att1])
audio_att2 = BatchNormalization()(audio_att2)

dropout_audio = Dropout(0.5)(audio_att2)
audio_prediction = Dense(5, activation='softmax')(dropout_audio)
audio_model = Model(inputs=audio_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=audio_input, outputs=audio_att2)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Text Branch
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_att = Attention(10, 20)([em_text, em_text, em_text])
text_att = BatchNormalization()(text_att)

text_att1 = Attention(10, 20)([text_att, text_att, text_att])
text_att1 = BatchNormalization()(text_att1)

text_att2 = Attention(10, 20)([text_att1, text_att1, text_att1])
text_att2 = BatchNormalization()(text_att2)

dropout_text = Dropout(0.5)(text_att)
text_prediction = Dense(5, activation='softmax')(dropout_text)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fusion Model
text_f_input = Input(shape=(200,))
audio_f_input = Input(shape=(64,))
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)
d_1 = Dense(200)(merge)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.6)(activation1)
d_2 = Dense(64)(d_drop1)
batch_nol2 = BatchNormalization()(d_2)
activation2 = Activation('relu')(batch_nol2)
d_drop2 = Dropout(0.6)(activation2)
f_prediction = Dense(5, activation='softmax')(d_drop2)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_inter_model = Model(inputs=[text_f_input, audio_f_input], outputs=merge)

text_acc = 0
train_text_inter = None
test_text_inter = None
for i in range(50):
    print('text branch, epoch: ', str(i))
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)

audio_acc = 0
train_audio_inter = None
test_audio_inter = None
for i in range(50):
    print('audio branch, epoch: ', str(i))
    audio_model.fit(train_audio_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate(test_audio_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict(train_audio_data, batch_size=batch_size)
        test_audio_inter = inter_audio_model.predict(test_audio_data, batch_size=batch_size)

final_acc = 0
result = None
for i in range(100):
    print('fusion branch, epoch: ', str(i))
    final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size,
                                         verbose=0)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= final_acc:
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)

r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)
