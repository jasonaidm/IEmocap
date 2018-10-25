from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from word2vec import embed_onehot, get_dictionary, initial_embed
import numpy as np
import random
import string
import scipy.io as scio
import os

dic_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/dic_mosi.txt'
label_category = ['1', '-1']
label_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/label_final.txt'
audio_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/Word_Mat_Processed/'
text_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/transcript.txt'
embed_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/'
visualization_text = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/Visualization/visualization_text.mat'
visualization_audio = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/Visualization/visualization_audio.mat'
visualization_fusion = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/Visualization/visualization_fusion.mat'
visualization_index = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/MOSI/Visualization/visualization_label.txt'
maxlen = 98
numclass = 2
num = 2199


def get_label(path):
    f = open(path, 'r')
    statistic = {'1': 0, '-1': 0}
    res = []
    for line in f:
        if line.split()[0] == label_category[0]:
            statistic[label_category[0]] += 1
            res.append(0)
        elif line.split()[0] == label_category[1]:
            statistic[label_category[1]] += 1
            res.append(1)
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
        if label[i] != 4:
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
    r_0 = {'0': 0, '1': 0}
    r_1 = {'0': 0, '1': 0}
    i = 0
    while i < len(test_label):
        if test_label[i] == 0:
            r_0[str(result[i])] += 1
        elif test_label[i] == 1:
            r_1[str(result[i])] += 1
        i += 1
    return r_0, r_1


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
    if result_fusion != []:
        if os.path.exists(visualization_fusion):
            os.remove(visualization_fusion)
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
