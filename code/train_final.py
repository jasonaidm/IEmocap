from __future__ import print_function
from self_attention import Attention, Position_Embedding
from load_final_data import get_data, analyze_data, data_generator, data_generator_output  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Embedding, concatenate, \
    GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, Lambda, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from keras import backend
from sklearn.utils import shuffle
from sklearn import preprocessing
from attention_model import AttentionLayer

max_features = 20000
batch_size = 16
epo = 100
filters = 128
flag = 0.60
numclass = 5
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

print('train_audio shape:', train_audio_data.shape)
print('train_text shape:', train_text_data.shape)
print('test_audio shape:', test_audio_data.shape)
print('test_text shape:', test_text_data.shape)
print('train_label shape:', train_label.shape)
print('test_label shape:', test_label.shape)


def weight_expand(x):
    return backend.expand_dims(x)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def weight_average(inputs):
    x = inputs[0]
    y = inputs[1]
    return (x + y) / 2


def data_normal(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x


# Audio branch
audio_input = Input(shape=(513, 64))
print('audio_input shape: ', audio_input.shape)
audio_att = Attention(4, 16)([audio_input, audio_input, audio_input])
audio_att = BatchNormalization()(audio_att)
audio_att = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att = BatchNormalization()(audio_att)
audio_att = Attention(4, 16)([audio_att, audio_att, audio_att])
audio_att = BatchNormalization()(audio_att)
audio_att = GlobalAveragePooling1D()(audio_att)
print('audio_att shape: ', audio_att.shape)
# frame_weight_exp = Lambda(weight_expand)(audio_att)
# frame_att = Lambda(weight_dot)([audio_input, frame_weight_exp])# ?,531,64 ?,531,1
# frame_att = Lambda(lambda x: backend.sum(x, axis=1))(frame_att)#
dropout_audio = Dropout(0.5)(audio_att)
model_frame = Model(audio_input, dropout_audio)

word_input = Input(shape=(98, 513, 64))
print('word_input shape: ', word_input.shape)
word_input = TimeDistributed(model_frame)(word_input)############
word_att = Attention(4, 16)([word_input, word_input, word_input])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])
word_att = BatchNormalization()(word_att)
word_att = Attention(4, 16)([word_att, word_att, word_att])
word_att = BatchNormalization()(word_att)
word_att = GlobalAveragePooling1D()(word_att)
print('word_att shape: ', word_att.shape)
# word_att = Lambda(weight_expand)(word_att) #,64,1  ,64
# word_attention = Lambda(weight_dot)([word_input, word_weight_exp])#,98,513,64 ,64,1
# word_att1 = Lambda(lambda x: backend.sum(x, axis=1))(word_attention)
dropout_word = Dropout(0.5)(word_att)

audio_prediction = Dense(5, activation='softmax')(dropout_word)
audio_model = Model(inputs=word_input, outputs=audio_prediction)#########
inter_audio_model = Model(inputs=word_input, outputs=[word_att])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Text Branch
text_input = Input(shape=(98,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_weight = Attention(10, 20)([em_text, em_text, em_text])
text_weight = BatchNormalization()(text_weight)
text_weight = Attention(10, 20)([text_weight, text_weight, text_weight])
text_weight = BatchNormalization()(text_weight)
text_weight = Attention(10, 20)([text_weight, text_weight, text_weight])
text_weight = BatchNormalization()(text_weight)
text_weight = GlobalAveragePooling1D()(text_weight)
print('text_weight shape: ', text_weight.shape)
# text_weight = Lambda(weight_expand)(text_weight)
# text_attention = Lambda(weight_dot)([em_text, text_weight_exp])
# text_att = Lambda(lambda x: backend.sum(x, axis=1))(text_attention)
dropout_text = Dropout(0.5)(text_weight)
text_prediction = Dense(5, activation='softmax')(dropout_text)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=[text_weight])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fusion Model
audio_f_input = Input(shape=(64,))  # 50，    #98,200      #98,
text_f_input = Input(shape=(200,))  # ，64 #98,200      #98,513,64
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.5)(merge)
print('merge shape: ', merge.shape)

'''
#merge_weight = AttentionLayer()(merge)
#merge_weight_exp = Lambda(weight_expand)(merge_weight)
#merge = Lambda(weight_dot)([merge, merge_weight_exp])
#merge = BatchNormalization()(merge)
#print('merge shape: ', merge.shape)
#merge = Attention(10,20)([merge,merge,merge])
#merge = BatchNormalization()(merge)
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
'''
d_1 = Dense(256)(merge)
batch_nol1 = BatchNormalization()(d_1)
activation1 = Activation('relu')(batch_nol1)
d_drop1 = Dropout(0.6)(activation1)
d_2 = Dense(128)(d_drop1)
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
for i in range(100):
    print('text branch, epoch: ', str(i))
    text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if acc_t >= text_acc:
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\text_model.h5')
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\text_model\\inter_text_model.h5')

train_audio_inter = None
test_audio_inter = None
audio_acc = 0
for i in range(50):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    audio_model.fit_generator(data_generator(audio_path, train_d, train_l, len(train_d)),
                              steps_per_epoch=len(train_d) / 4, epochs=1, verbose=1)
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / 4)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\audio_model.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model.h5')
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label,
                                  len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label,
                                  len(test_audio_data)),
            steps=len(test_audio_data))
'''
inter_audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_model\inter_audio_model1.h5')
train_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, train_audio_data, train_label,
                                                                              len(train_audio_data)),
                                                        steps=len(train_audio_data))
test_audio_inter = inter_audio_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_label,
                                                                             len(test_audio_data)),
                                                       steps=len(test_audio_data))
inter_text_model.load_weights(r'E:\Yue\Code\ACL_entire\text_model\inter_text_model.h5')
train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
'''
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
        final_model.save_weights(r'E:\Yue\Code\ACL_entire\final_model\final_model.h5')
        final_inter_model.save_weights(r'E:\Yue\Code\ACL_entire\final_model\final_inter_model.h5')
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        test_fusion_weight = final_inter_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)

r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)
