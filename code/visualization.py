from sklearn import preprocessing
from keras.preprocessing import sequence
import scipy.io as scio
import numpy as np


text_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/visualization_text.mat'
audio_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/visualization_audio.mat'
fusion_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/visualization_fusion.mat'
index_path = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/visualization_label.txt'
out_txt = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/result/text.txt'
out_aud = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Average_4_Category/result/audio.txt'
out_fus = '/media/yue/2e423a78-12f5-4de2-b748-381c1cede85f/ACL_2018/Visualization/Attention_4_Category/result/fusion.txt'

text_data = scio.loadmat(text_path)
text_data = text_data['visulization']

audio_data = scio.loadmat(audio_path)
audio_data = audio_data['visulization']

#fusion_data = scio.loadmat(fusion_path)
#fusion_data = fusion_data['visulization']

#audio_data = preprocessing.scale(audio)
#text_data = preprocessing.scale(text)

f = open(out_txt, 'a')
for i in range(len(text_data)):
    for j in text_data[i]:
        f.write(str(j)+' ')
    f.write('\n')
f.close()

t = open(out_aud, 'a')
for i in range(len(audio_data)):
    for j in audio_data[i]:
        t.write(str(j)+' ')
    t.write('\n')
t.close()

"""
k = open(out_fus, 'a')
for i in range(len(fusion_data)):
    for j in fusion_data[i]:
        k.write(str(j)+' ')
    k.write('\n')
k.close()
"""
