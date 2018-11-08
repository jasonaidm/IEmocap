from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from word2vec import embed_onehot, get_dictionary, initial_embed
import numpy as np
import random
import string
import scipy.io as scio
import os

dic_path = r'E:/Yue/Entire Data/ACL_2018/dictionary.txt'
label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu']
label_path = r'E:/Yue/Entire Data/ACL_2018/label_output.txt'
audio_path = r'E:/Yue/Entire Data/IEMOCAP/New_Channel_1_Nor/'
# hier_audio_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018/Word Mat Norm_2/'
text_path = r'E:/Yue/Entire Data/ACL_2018/text_output.txt'
embed_path = r'E:/Yue/Entire Data/ACL_2018/'
maxlen = 50
numclass = 4  # 3


def get_label(path):  # numclass = 3
    f = open(path, 'r')
    statistic = {'ang': 0, 'exc': 0, 'sad': 0, 'fru': 0, 'hap': 0, 'neu': 0}
    res = []
    for line in f:
        if line.split()[0] == label_category[0]:
            statistic[label_category[0]] += 1
            res.append(0)
        elif line.split()[0] == label_category[1]:
            statistic[label_category[1]] += 1
            res.append(1)
        elif line.split()[0] == label_category[2]:
            statistic[label_category[2]] += 1
            res.append(2)
        elif line.split()[0] == label_category[3]:
            statistic[label_category[3]] += 1
            res.append(3)
        elif line.split()[0] == label_category[4]:
            statistic[label_category[4]] += 1
            res.append(1)
        elif line.split()[0] == label_category[5]:
            statistic[label_category[5]] += 1
            res.append(3)  # append(4)
    print(statistic)
    return res


def get_mat_data(path):
    res = []
    i = 0
    while i < 7204:#3793
        tmp = scio.loadmat(path + str(i) + ".mat")
        tmp = tmp['z1']
        tmp = sequence.pad_sequences(tmp, padding='post', truncating='post', dtype='float32', maxlen=2250)
        # print(tmp.shape)
        tmp = tmp.transpose()
        res.append(tmp)
        i += 1
    # res = np.array(res)
    return res


def get_hier_mat_data():
    res = []
    i = 0
    while i < 3793:
        res.append(i)
        i += 1
    return res


def get_text_data(path, dic):
    f = open(path, 'r')
    res = []
    i = 0
    for line in f:
        text = embed_onehot(dic, line.translate(str.maketrans('', '', string.punctuation)))
        res.append(text)
        i += 1
    f.close()
    return res


# label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu']
def seprate_by_emotion(path, data):
    f = open(path, 'r')
    ang = []
    hap_exc = []
    neu_sad = []
    fru = []
    i = 0
    for line in f:
        if line.split()[0] == label_category[0]:
            ang.append(data[i])
        elif line.split()[0] == label_category[1]:
            hap_exc.append(data[i])
        elif line.split()[0] == label_category[2]:
            neu_sad.append(data[i])
        elif line.split()[0] == label_category[3]:
            fru.append(data[i])
        elif line.split()[0] == label_category[4]:
            hap_exc.append(data[i])
        elif line.split()[0] == label_category[5]:
            neu_sad.append(data[i])  # append(4)
        i += 1
    return ang, hap_exc, neu_sad, fru

'''
def seperate_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label = [], []
    i = 0
    while i < len(audio_data):
        if random.randint(0, 100) < 80:
            train_audio_data.append(audio_data[i])
            train_text_data.append(text_data[i])
            train_label.append(label[i])
        else:
            test_audio_data.append(audio_data[i])
            test_text_data.append(text_data[i])
            test_label.append(label[i])
        i += 1
    return np.array(train_audio_data), train_text_data, train_label, np.array(test_audio_data), test_text_data, test_label
'''
def seperate_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label = [], []
    ang_text, hap_exc_text, neu_sad_text, fru_text = seprate_by_emotion(text_data)
    ang_audio, hap_exc_audio, neu_sad_audio, fru_audio = seprate_by_emotion(audio_data)
    ang_label, hap_exc_label, neu_sad_label, fru_label = seprate_by_emotion(label)
    ang_i = 0
    while ang_i < len(ang_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(ang_text[ang_i])
            train_audio_data.append(ang_audio[ang_i])
            train_label.append(ang_label[ang_i])
        else:
            test_text_data.append(ang_text[ang_i])
            test_audio_data.append(ang_audio[ang_i])
            test_label.append(ang_label[ang_i])
        ang_i += 1

    hap_exc_i = 0
    while hap_exc_i < len(hap_exc_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(hap_exc_text[hap_exc_i])
            train_audio_data.append(hap_exc_audio[hap_exc_i])
            train_label.append(hap_exc_label[hap_exc_i])
        else:
            test_text_data.append(hap_exc_text[hap_exc_i])
            test_audio_data.append(hap_exc_audio[hap_exc_i])
            test_label.append(hap_exc_label[hap_exc_i])
        hap_exc_i += 1

    neu_sad_i = 0
    while neu_sad_i < len(neu_sad_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(neu_sad_text[neu_sad_i])
            train_audio_data.append(neu_sad_audio[neu_sad_i])
            train_label.append(neu_sad_label[neu_sad_i])

        else:
            test_text_data.append(neu_sad_text[neu_sad_i])
            test_audio_data.append(neu_sad_audio[neu_sad_i])
            test_label.append(neu_sad_label[neu_sad_i])
        neu_sad_i += 1

    fru_i = 0
    while fru_i < len(fru_audio):
        # ang data
        if random.randint(0, 100) < 80:
            train_text_data.append(fru_text[fru_i])
            train_audio_data.append(fru_audio[fru_i])
            train_label.append(fru_label[fru_i])

        else:
            test_text_data.append(fru_text[fru_i])
            test_audio_data.append(fru_audio[fru_i])
            test_label.append(fru_label[fru_i])
        fru_i += 1

    return np.array(train_audio_data), train_text_data, train_label, np.array(
        test_audio_data), test_text_data, test_label


def seperate_hier_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label = [], []
    i = 0
    while i < len(audio_data):
        if random.randint(0, 100) < 80:
            train_audio_data.append(audio_data[i])
            train_text_data.append(text_data[i])
            train_label.append(label[i])
        else:
            test_audio_data.append(audio_data[i])
            test_text_data.append(text_data[i])
            test_label.append(label[i])
        i += 1
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label


def analyze_data(test_label, result):  # numcalss = 4
    r_0 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_1 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_2 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_3 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    r_4 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    i = 0
    while i < len(test_label):  # 4
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


def train_data_generation(audio_data, text_data, label):
    i = 0
    r_audio, r_text, r_label = [], [], []
    while i < len(audio_data):
        if label[i] == 1:
            if random.randint(0, 100) < 75:
                r_audio.append(audio_data[i])
                r_text.append(text_data[i])
                r_label.append(label[i])
        elif label[i] == 3:
            if random.randint(0, 100) < 40:
                r_audio.append(audio_data[i])
                r_text.append(text_data[i])
                r_label.append(label[i])
        else:
            r_audio.append(audio_data[i])
            r_text.append(text_data[i])
            r_label.append(label[i])
        i += 1
    return np.array(r_audio), r_text, r_label


def data_generator(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        j = 0
        while j < 8:
            if i == num:
                i = 0
            tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
            tmp = tmp['z1']
            res.append(tmp)
            res_label.append(audio_label[i])
            j += 1
            i += 1
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=98)
        yield (np.array(res), np.array(res_label))


def data_generator_output(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        if i == num:
            i = 0
        tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
        tmp = tmp['z1']
        res.append(tmp)
        res_label.append(audio_label[i])
        i += 1
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=98)
        yield (np.array(res), np.array(res_label))


"""

def data_generator(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        if i == num:
            i = 0
        tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
        tmp = tmp['z1']
        res.append(tmp)
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=98)
        #print(res.shape)
        res_label.append(audio_label[i])
        i += 1
        yield (np.array(res), np.array(res_label))


def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_mat_data(audio_path)
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o = seperate_dataset(audio_data, text_data, label)
    test_label = to_categorical(test_label_o, num_classes=4)
    test_text_data = sequence.pad_sequences(test_text_data, maxlen=50)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic


def process_train_data(train_audio, train_text, train_label):
    train_label = to_categorical(train_label, num_classes=4)
    train_text = sequence.pad_sequences(train_text, maxlen=50)
    return train_audio, train_text, train_label

"""


def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_mat_data(audio_path)
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o = seperate_dataset(
        audio_data, text_data, label)
    train_label = to_categorical(train_label,
                                 num_classes=numclass)  # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
    train_text_data = sequence.pad_sequences(train_text_data, maxlen=maxlen)  # to 2D numpy array
    test_label = to_categorical(test_label_o, num_classes=numclass)  # 3
    test_text_data = sequence.pad_sequences(test_text_data, maxlen=maxlen)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic


"""
def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_hier_mat_data()
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o = seperate_hier_dataset(audio_data, text_data, label)
    train_label = to_categorical(train_label, num_classes=numclass)
    train_text_data = sequence.pad_sequences(train_text_data, maxlen=maxlen)
    test_label = to_categorical(test_label_o, num_classes=numclass)
    test_text_data = sequence.pad_sequences(test_text_data, maxlen=maxlen)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic
"""
