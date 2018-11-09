from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from attention_model import AttentionLayer


def hierarchical_attention(max_seq, emb_weights=None, embedding_size=None, vocab_size=None,  # embedding
                           recursive_class=GRU, word_rnnsize=100,  # rnn
                           drop_wordemb=0.2, drop_wordrnnout=0.2):
    """
    Creates a model based on the Hierarchical Attention model according to : https://arxiv.org/abs/1606.02393
    inputs:
        maxSeq : max size for sentences
        embedding
            embWeights : numpy matrix with embedding values
            embeddingSize (if embWeights is None) : embedding size
            vocabSize (if embWeights is None) : vocabulary size
        Recursive Layers
            recursiveClass : class for recursive class. Default is GRU
            wordRnnSize : RNN size for word sequence
            sentenceRnnSize :  RNN size for sentence sequence
        Dense Layers
            wordDenseSize: dense layer at exit from RNN , on sentence at word level
            sentenceHiddenSize : dense layer at exit from RNN , on document at sentence level
        Dropout

    returns : Two models. They are the same, but the second contains multiple outputs that can be use to analyse attention.
    """

    # Sentence level logic

    # Input Layer
    words_inputs = Input(shape=(max_seq,), dtype='int32', name='words_input')

    # Word embedding layer
    if emb_weights is None:
        emb = Embedding(vocab_size, embedding_size, mask_zero=True)(words_inputs)
    else:
        emb = Embedding(emb_weights.shape[0], emb_weights.shape[1], mask_zero=True, weights=[emb_weights],
                        trainable=False)(words_inputs)
    """
    if drop_wordemb != 0.0:
        emb = Dropout(drop_wordemb)(emb)
    """
    # RNN layer (GRU/LSTM/biLSTM)
    word_rnn = Bidirectional(recursive_class(word_rnnsize, return_sequences=True), merge_mode='concat')(emb)
    # word_rnn = BatchNormalization()(word_rnn)

    if drop_wordrnnout > 0.0:
        word_rnn = Dropout(drop_wordrnnout)(word_rnn)

    sentence_att = AttentionLayer()(word_rnn)


    sentence_out = Dense(6, activation="softmax", name="words_Out")(sentence_att)

    model = Model(words_inputs, sentence_out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    """
    documentInputs = Input(shape=(None, maxSeq), dtype='int32', name='document_input')
    sentenceMasking = Masking(mask_value=0)(documentInputs)
    sentenceEmbbeding = TimeDistributed(modelSentence)(sentenceMasking)
    sentenceAttention = TimeDistributed(modelSentAttention)(sentenceMasking)
    sentenceRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(
        sentenceEmbbeding)
    if dropSentenceRnnOut > 0.0:
        sentenceRnn = Dropout(dropSentenceRnnOut)(sentenceRnn)
    attentionSent = AttentionLayer()(sentenceRnn)

    documentEmb = merge([sentenceRnn, attentionSent], mode=lambda x: x[1] * x[0], output_shape=lambda x: x[0])
    documentEmb = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: (x[0], x[2]), name="att2")(documentEmb)
    documentOut = Dense(1, activation="sigmoid", name="documentOut")(documentEmb)

    model = Model(input=[documentInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    modelAttentionEv = Model(input=[documentInputs], output=[documentOut, sentenceAttention, attentionSent])
    modelAttentionEv.compile(loss='binary_crossentropy',
                             optimizer='rmsprop',
                             metrics=['accuracy'])
    """
    return model
