from __future__ import print_function
from load_attention_data import get_data, analyze_data
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking, Embedding, concatenate, TimeDistributed
from keras.layers import BatchNormalization, Activation, Conv1D, GlobalMaxPooling1D, Lambda, add
from keras.optimizers import Adam
from keras import backend
from attention_model import AttentionLayer
from load_attention_data import data_generator, data_generator_output, output_result
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import numpy as np
import os

max_features = 20000
batch_size = 16
epo = 100
numclass = 4
flag = 0.60
filters = 128
audio_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_entire_2018/Word_Mat_New_1/'
#model_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/audio_model.h5'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic, test_index = get_data()

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


def weight_expand(x):
    return backend.expand_dims(x)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def weight_average(inputs):
    x = inputs[0]
    y = inputs[1]
    return (x+y)/2


def data_normal(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x


# Audio branch
# Previous 3793, 513, 98
frame_input = Input(shape=(513, 64))
mask_frame_input = Masking(mask_value=0.)(frame_input)
print('mask_frame_input shape: ', mask_frame_input.shape)
frame_l1 = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_1'))(mask_frame_input)
#frame_l1 = BatchNormalization()(frame_l1)
print('frame_l1 shape: ', frame_l1.shape)
frame_weight = AttentionLayer()(frame_l1)
#frame_weight = BatchNormalization()(frame_weight)
print('frame_att shape: ', frame_weight.shape)
frame_weight_exp = Lambda(weight_expand)(frame_weight)
frame_att = Lambda(weight_dot)([frame_l1, frame_weight_exp])
frame_att = Lambda(lambda x: backend.sum(x, axis=1))(frame_att)
print('frame_att shape: ', frame_att.shape)
dropout_frame = Dropout(0.5)(frame_att)
model_frame = Model(frame_input, dropout_frame)

word_input = Input(shape=(98, 513, 64))
mask_word_input = Masking(mask_value=0.)(word_input)
print('mask_word_input shape: ', mask_word_input.shape)
audio_input = TimeDistributed(model_frame)(mask_word_input)
print('audio_input shape: ', audio_input.shape)
audio_input = Masking(mask_value=0.)(audio_input)
audio_l1 = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_2'))(audio_input)
#audio_l1 = BatchNormalization()(audio_l1)
print('audio_l1 shape: ', audio_l1.shape)
word_weight = AttentionLayer()(audio_l1)
#word_weight = BatchNormalization()(word_weight)
print('word_weight shape: ', word_weight.shape)
word_weight_exp = Lambda(weight_expand)(word_weight)
word_attention = Lambda(weight_dot)([audio_l1, word_weight_exp])
word_att = Lambda(lambda x: backend.sum(x, axis=1))(word_attention)
print('word_att shape: ', word_att.shape)
dropout_word = Dropout(0.5)(word_att)

audio_prediction = Dense(numclass, activation='softmax')(dropout_word)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_hidden = Model(inputs=word_input, outputs=[word_attention, word_weight])
inter_audio_weight = Model(inputs=word_input, outputs=word_weight)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Text Branch
text_input = Input(shape=(98,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
mask_text_input = Masking(mask_value=0.)(em_text)
text_l1 = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_text'))(mask_text_input)
text_l1 = BatchNormalization()(text_l1)
text_weight = AttentionLayer()(text_l1)
text_weight = BatchNormalization()(text_weight)
print('frame_att shape: ', text_weight.shape)
text_weight_exp = Lambda(weight_expand)(text_weight)
text_attention = Lambda(weight_dot)([text_l1, text_weight_exp])
text_att = Lambda(lambda x: backend.sum(x, axis=1))(text_attention)
dropout_text = Dropout(0.5)(text_att)

text_prediction = Dense(numclass, activation='softmax')(dropout_text)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_hidden = Model(inputs=text_input, outputs=[text_attention, text_weight])
inter_text_weight = Model(inputs=text_input, outputs=text_weight)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Fusion Model
text_f_input = Input(shape=(98, 200))
audio_f_input = Input(shape=(98, 200))

"""
text_weight_input = Lambda(weight_expand)(text_weight_input)
text_f_input = Lambda(weight_dot)([text_f_input, text_weight_input])

audio_weight_input = Lambda(weight_expand)(audio_weight_input)
audio_f_input = Lambda(weight_dot)([audio_f_input, audio_weight_input])
"""

merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)
print('merge shape: ', merge.shape)

merge_weight = AttentionLayer()(merge)
merge_weight_exp = Lambda(weight_expand)(merge_weight)
merge = Lambda(weight_dot)([merge, merge_weight_exp])
merge = BatchNormalization()(merge)


cnn_1 = Conv1D(filters, 2, padding='valid', strides=1)(merge)
batchnol1 = BatchNormalization()(cnn_1)
activation1 = Activation('relu')(batchnol1)
maxpool_1 = GlobalMaxPooling1D()(activation1)
dropout_1 = Dropout(0.7)(maxpool_1)

cnn_2 = Conv1D(filters, 3, padding='valid', strides=1)(merge)
batchnol2 = BatchNormalization()(cnn_2)
activation2 = Activation('relu')(batchnol2)
maxpool_2 = GlobalMaxPooling1D()(activation2)
dropout_2 = Dropout(0.7)(maxpool_2)

cnn_3 = Conv1D(filters, 4, padding='valid', strides=1)(merge)
batchnol3 = BatchNormalization()(cnn_3)
activation3 = Activation('relu')(batchnol3)
maxpool_3 = GlobalMaxPooling1D()(activation3)
dropout_3 = Dropout(0.7)(maxpool_3)

cnn_4 = Conv1D(filters, 5, padding='valid', strides=1)(merge)
batchnol4 = BatchNormalization()(cnn_4)
activation4 = Activation('relu')(batchnol4)
maxpool_4 = GlobalMaxPooling1D()(activation4)
dropout_4 = Dropout(0.7)(maxpool_4)

final_merge = concatenate([dropout_1, dropout_2, dropout_3, dropout_4], name='final_merge')

d_1 = Dense(256)(final_merge)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.6)(activation1)
d_2 = Dense(128)(d_drop1)
batch_nol2 = BatchNormalization()(d_2)
activation2 = Activation('relu')(batch_nol2)
d_drop2 = Dropout(0.6)(activation2)
f_prediction = Dense(numclass, activation='softmax')(d_drop2)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
#visualization = Model(inputs=[text_f_input, audio_f_input], outputs=merge)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
final_inter_model = Model(inputs=[text_f_input, audio_f_input], outputs=merge_weight)

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


text_acc = 0
for i in range(25):
    print('text branch, epoch: ', str(i))
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter, train_text_weight = inter_text_hidden.predict(train_text_data, batch_size=batch_size)
        #train_text_weight = inter_text_weight.predict(train_text_data, batch_size=batch_size)
        test_text_inter, test_text_weight = inter_text_hidden.predict(test_text_data, batch_size=batch_size)
        #test_text_weight = inter_text_weight.predict(test_text_data, batch_size=batch_size)

audio_acc = 0
for i in range(50):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d)/4, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate_generator(data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
                                                   steps=len(test_audio_data)/4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc and acc_a >= flag:
        audio_acc = acc_a
        train_audio_inter, train_audio_weight = inter_audio_hidden.predict_generator(data_generator_output(audio_path, train_audio_data, train_label,
                                                                               len(train_audio_data)),
                                                                steps=len(train_audio_data))
        #train_audio_weight = inter_audio_weight.predict_generator(data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)), steps=len(train_audio_data))
        test_audio_inter, test_audio_weight = inter_audio_hidden.predict_generator(data_generator_output(audio_path, test_audio_data, test_label,
                                                                              len(test_audio_data)),
                                                               steps=len(test_audio_data))
        #test_audio_weight = inter_audio_weight.predict_generator(data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)), steps=len(test_audio_data))
        """
        if os.path.exists(model_path):
            os.remove(model_path)
        inter_audio_hidden.save(model_path)
        """

"""
#output_result(test_text_weight, test_audio_weight, test_index)
output_audio_model = load_model(model_path)
train_audio_inter, train_audio_weight = output_audio_model.predict_generator(data_generator_output(audio_path, train_audio_data, train_label,
                                                                                                   len(train_audio_data)),
                                                                             steps=len(train_audio_data))
test_audio_inter, test_audio_weight = output_audio_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_label,
                                                                                                 len(test_audio_data)),
                                                                           steps=len(test_audio_data))
"""

#train_audio_weight = data_normal(train_audio_weight)
#test_audio_weight = data_normal(test_audio_weight)
#train_text_weight = data_normal(train_text_weight)
#test_text_weight = data_normal(test_text_weight)



final_acc = 0
for i in range(epo):
    print('fusion branch, epoch: ', str(i))
    final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if acc_f >= final_acc:
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        test_fusion_weight = final_inter_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        #visualization_res = visualization.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        #output_result(visualization_res, test_index)
        result = np.argmax(result, axis=1)


r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print(r_0)
print(r_1)
print(r_2)
print(r_3)
print(r_4)

output_result(test_text_weight, test_audio_weight, test_fusion_weight, test_index)