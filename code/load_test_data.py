from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from word2vec import embed_onehot, get_dictionary, initial_embed
import numpy as np
import random
import string
import scipy.io as scio
import os

dic_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/dictionary_new.txt'
label_category = ['neu', 'hap', 'sad', 'ang', 'oth', 'fru', 'exc', 'fea', 'sur', 'dis']
label_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACMMM_2018/IEMOCAP/label.txt'
audio_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACMMM_2018/IEMOCAP/Word_MFSC/'
text_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACMMM_2018/IEMOCAP/transcription.txt'
embed_path = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/'
visualization_text = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/Visualization/visualization_text.mat'
visualization_audio = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/Visualization/visualization_audio.mat'
visualization_fusion = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/Visualization/visualization_fusion.mat'
visualization_index = '/media/yeu/cdfd566c-2b64-486d-ac81-81c7dedfd5df/ACL_2018_entire/Visualization/visualization_label.txt'
maxlen = 100
numclass = 10
num = 10039


def get_label(path):
    f = open(path, 'r')
    statistic = {'neu':0, 'hap':0, 'sad':0, 'ang':0, 'oth':0, 'fru':0, 'exc':0, 'fea':0, 'sur':0, 'dis':0}
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
            res.append(4)
        elif line.split()[0] == label_category[5]:
            statistic[label_category[5]] += 1
            res.append(5)
        elif line.split()[0] == label_category[6]:
            statistic[label_category[6]] += 1
            res.append(6)
        elif line.split()[0] == label_category[7]:
            statistic[label_category[7]] += 1
            res.append(7)
        elif line.split()[0] == label_category[8]:
            statistic[label_category[8]] += 1
            res.append(8)
        elif line.split()[0] == label_category[9]:
            statistic[label_category[9]] += 1
            res.append(9)
    print(statistic)
    return res


def get_hier_mat_data():
    res = []
    i = 0
    while i < num:
        res.append(i)
        i += 1
    return res



def get_text_data(path, dic):
    f = open(path, 'r')
    res = []
    i = 0
    for line in f:
        text = embed_onehot(dic, line.translate(None, string.punctuation))
        res.append(text)
        i += 1
    f.close()
    return res


def seperate_hier_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label, test_index = [], [], []
    i = 0
    while i < len(audio_data):
        # remove the following line to change back to 5 category
        if random.randint(0, 100) < 80:
            train_audio_data.append(audio_data[i])
            train_text_data.append(text_data[i])
            train_label.append(label[i])
        else:
            test_audio_data.append(audio_data[i])
            test_text_data.append(text_data[i])
            test_label.append(label[i])
            test_index.append(i)
        i += 1
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_index



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


def data_generator(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        j = 0
        while j < 4:
            if i == num:
                i = 0
            tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
            tmp = tmp['z1']
            res.append(tmp)
            res_label.append(audio_label[i])
            j += 1
            i += 1
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=maxlen)
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
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=maxlen)
        yield (np.array(res), np.array(res_label))


def output_result(result_text, result_audio, result_fusion, index):
    if os.path.exists(visualization_text):
        os.remove(visualization_text)
    scio.savemat(visualization_text, {'visulization': result_text})
    if os.path.exists(visualization_audio):
        os.remove(visualization_audio)
    scio.savemat(visualization_audio, {'visulization': result_audio})
    if os.path.exists(visualization_fusion):
        os.remove(visualization_fusion)
    if result_fusion != []:
        scio.savemat(visualization_fusion, {'visulization': result_fusion})
    if os.path.exists(visualization_index):
        os.remove(visualization_index)
    f = open(visualization_index, 'a')
    for i in range(len(index)):
        f.write(str(index[i])+'\n')
    f.close()


def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_hier_mat_data()
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o, test_index = seperate_hier_dataset(audio_data, text_data, label)
    train_label = to_categorical(train_label, num_classes=numclass)
    train_text_data = sequence.pad_sequences(train_text_data, padding='post', truncating='post', maxlen=maxlen)
    test_label = to_categorical(test_label_o, num_classes=numclass)
    test_text_data = sequence.pad_sequences(test_text_data, padding='post', truncating='post', maxlen=maxlen)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic, test_index
