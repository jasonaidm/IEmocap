# Train result: 11.17 (4->5 class) 
* 1000 epoches for fusion and each branch
* Final result:   
* text acc:  0.5700066357396182  audio acc:  0.4844061048836129  final acc:  0.6270736562707365  
* 0 {'0': 157, '1': 8, '2': 6, '3': 49, '4': 14}  
* 1 {'0': 23, '1': 228, '2': 14, '3': 15, '4': 41}  
* 2 {'0': 3, '1': 12, '2': 116, '3': 24, '4': 36}    
* 3 {'0': 47, '1': 18, '2': 21, '3': 259, '4': 52}  
* 4 {'0': 13, '1': 55, '2': 30, '3': 81, '4': 185}  
# text branch :
* 1.去掉标点符号 -> embed_onehot 根据dic编号 加载text数组-> pad_sequences-> seperate_dataset
* 2.onehot_vector（去标点，将词语贴在一起） -> build_dataset（为每一个单词标号） ->save_dictionary
audio branch:(下载下来的时候是一维feature矩阵)
Word_Mat_01_original 找到原本的mat文件（找出来看下，貌似是一维的）
pad_sequences 一维转二维 （该函数将一个 num_samples 的序列（整数列表）转化为一个 2D Numpy 矩阵，其尺寸为
(num_samples, num_timesteps)。 num_timesteps 要么是给定的 maxlen 参数，要么是最长序列的长度。）
savemat
normalization
seperate_dataset

# label：
* to_categorical（将类向量（整数）转换为二进制类矩阵。）->seperate_dataset


# datasets:
## text:
* text_output_new 7204 sentenses


## audio:
* max 64*2219
* colomn: length
* line: 64 windos size

## label:
* label_output_new 7204 emotions(6 kinds)

## emben word2vector:
* glove.6B.50d
* initial_embed具体看一下

## dictionary:
* text转词典
