def get_train_test():
    import os
    os.chdir("/Users/huashang/Documents/DeepLearning/ACL_2018/IEMOCAP/")

    text_file = open("text_output.txt", "r")
    lines = text_file.readlines()
    li = len(lines)

    dict = {}
    text_file_1 = open("dictionary.txt", "r")
    row = text_file_1.read().split('\n')[:-1]
    '''print(row)'''
    for i in range(0, len(row)):
        roow = row[i].split()
        dict1 = {roow[0]: roow[1]}
        dict.update(dict1)

    '''print(dict)'''

    import string
    from nltk.tokenize import RegexpTokenizer

    # get training data(including X_train, X_test)
    train = [[] for m in range(li)]
    maxlen = []
    for j in range(0, li):
        var1 = lines[j]
        tokenizer = RegexpTokenizer(r'\w+')
        token = tokenizer.tokenize(var1.translate(None, string.punctuation))
        '''print('token:',token)'''
        maxlen.append(len(token))
        for k in range(0, len(token)):
            if token[k] in dict:
                train[j].append(int(dict[token[k]]))
            else:
                train[j].append(int(0))

    '''print('train:',train)'''
    # cut texts after this number of words (among top max_features most common words)
    maxlen = sorted(maxlen)
    maxlength = maxlen[int(len(maxlen)*0.99)]
    print('maxlength: ',maxlength)


    text_file = open("label_output.txt", "r")
    lines = text_file.read().split('\n')[:-1]
    '''print(lines)'''
    list_2 = list(set(lines))
    '''print(list_2)'''

    dict2 = {}
    for i in range(len(list_2)):
        dict3 = {list_2[i]: i}
        dict2.update(dict3)

    '''print(dict2)'''

    test = []
    for i in range(len(lines)):
        if lines[i] in dict2:
            test.append(dict2[lines[i]])

    '''print("test: ",test)'''

    index = []
    for i in range(li):
        index.append(i)

    import random
    lii = int(li * 0.8)
    #index_train = random.sample(range(li), lii)
    index_train = []
    for i in range(lii):
        index_train.append(i)
    index_test = list(set(index).difference(set(index_train)))
    #print("index_train=",index_train)
    #print("index_test=", index_test)
    X_train = []
    Y_train = []
    print("**************************")
    for j in index_train:
        X_train.append(train[j])
        Y_train.append(test[j])

    print(len(X_train), "X_train")
    print(len(Y_train), "Y_train")

    X_test = []
    Y_test = []

    '''print("index_test:",index_test)'''
    for j in index_test:
        X_test.append(train[j])
        Y_test.append(test[j])

    print(len(X_test), "X_test")
    print(len(Y_test), "Y_test")
    print("**************************")
    print("Preprocessing Finished")

    return (X_train, Y_train), (X_test, Y_test), maxlength
