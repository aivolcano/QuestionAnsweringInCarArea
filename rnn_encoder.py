import tensorflow as tf
from keras.layers import Embedding
import gensim
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2  #整数除法
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        
        word2vec_path = 'F:/7.NLP导师名企班/utils/w2v.bin'
        Word2VecModel = gensim.model.Word2Vec.load(word2vec_path)
        vector = Word2VecModel.wv['空间'].shape

        vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]#储存所有的词语

        word_index = {" ": 0} #初始化 [word: token],后期tokenize语料库就是用该词典
        word_vector = {}    #初始化[word: vector]字典
        #  初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
        # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
        embedding_matrix = np.zeros(len(vocab_list)+1, Word2VecModel.vector_size)
        # gensim的word2vec模型 把所有的单词和 词向量 都存储在了Word2VecModel.wv里面，讲道理直接使用这个.wv即可。

        # 填充字典和矩阵
        for i in range(len(vocab_list)):
            word = vocab_list[i]
            word_index[word] = i+1
            word_vector[word] = Word2VecModel.wv[word]
            embeddings_matrix[i+1] = Word2VecModel.wv[word]

        # 在keras中定义Embedding层 并使用预训练的词向量
        EMBEDDING_DIM = 256 #词向量的维度 之前是256
        MAX_SEQENCE_LENGTH = 10

        self.embedding = tf.keras.layers.Embedding(input_dim = len(embeddings_matrix),#字典长度
                                                    EMBEDDING_DIM = EMBEDDING_DIM, #词向量长度
                                                    weights = [embeddings_matrix], #预训练词向量的系数（重点）
                                                    input_length = MAX_SEQENCE_LENGTH, #每句话的最大长度（必须padding）
                                                    trainable = False #训练过程中 是否更新词向量            
        )                           

        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.lstm = tf.keras.layers.LSTM(256,return_sequences=True, return_state=True,recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRU(256,return_sequences=True, return_state=True,recurrent_initializer='glorot_uniform') #https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRU
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
