from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, Masking
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from attention_model import AttentionLayer
import numpy as np
import random
import scipy.io as scio

label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu']
label_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/label_output.txt'
audio_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Audio Mat Norm_2/'

def get_label(path):
    f = open(path, 'r')
    res = []
    for line in f:
        if line.split()[0] == label_category[0]:
            res.append(0)
        elif line.split()[0] == label_category[1]:
            res.append(1)
        elif line.split()[0] == label_category[2]:
            res.append(2)
        elif line.split()[0] == label_category[3]:
            res.append(4)
        elif line.split()[0] == label_category[4]:
            res.append(1)
        elif line.split()[0] == label_category[5]:
            res.append(3)
    return res


def get_mat_data(path):
    res = []
    i = 0
    while i < 3793:
        tmp = scio.loadmat(path+str(i)+".mat")
        tmp = tmp['z1']
        tmp = sequence.pad_sequences(tmp, padding='post', truncating='post', dtype='float32', maxlen=2250)
        #print(tmp.shape)
        tmp = tmp.transpose()
        res.append(tmp)
        i += 1
    #res = np.array(res)
    return res

def seperate_dataset(data, label):
    train_data, test_data = [], []
    train_label, test_label = [], []
    i = 0
    while i < len(data):
        if random.randint(0, 100) < 80:
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
        i += 1
    return np.array(train_data), train_label, np.array(test_data), test_label


def analyze_data(test_label, result):
    r_0 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_1 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_2 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_3 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_4 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    i = 0
    while i < len(test_label):
        if test_label[i] == 0:
            r_0[str(result[i])] += 1
        elif test_label[i] == 1:
            r_1[str(result[i])] += 1
        elif test_label[i] == 2:
            r_2[str(result[i])] += 1
        elif test_label[i] == 3:
            r_3[str(result[i])] += 1
        elif test_label[i] == 4:
            r_4[str(result[i])] += 1
        i += 1
    return r_0, r_1, r_2, r_3, r_4

# Audio BLSTM

audio_input = Input(shape=(2250, 64))
mask_input = Masking(mask_value=0.)(audio_input)
audio_l1 = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.25, name='LSTM_1'))(mask_input)
#audio_l2 = Bidirectional(LSTM(256, return_sequences=False, recurrent_dropout=0.5, name='LSTM_2'))(audio_l1)
audio_att = AttentionLayer()(audio_l1)
activation5 = Dropout(0.25)(audio_att)


final_prediction = Dense(5, activation='softmax')(activation5)
final_model = Model(inputs=audio_input, outputs=final_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


label = get_label(label_path)
data = get_mat_data(audio_path)
train_data, train_label, test_data, test_label_o = seperate_dataset(data, label)
test_label = to_categorical(test_label_o, num_classes=5)
train_label = to_categorical(train_label, num_classes=5)
print('train_data', train_data.shape, train_label.shape)
print('test_data', test_data.shape, test_label.shape)

final_model.fit(train_data, train_label, batch_size=16, epochs=100, verbose=1,
                validation_data=(test_data, test_label))
result = final_model.predict(test_data, batch_size=32)
result = np.argmax(result, axis=1)
#print(result)
print(len(result))

r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)

print(r_0)
print(r_1)
print(r_2)
print(r_3)
print(r_4)