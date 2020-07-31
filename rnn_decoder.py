import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output): #计算e_ij； 用weighted normalizatione_ij归一化处理
        """
        :param dec_hidden: shape=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)  # shape=(16, 1, 256) 加一层
        # att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        """
        定义score
        """
        #应该是用tanh加一层非线性层
        score = self.V(tf.keras.activations.tanh(self.W1(enc_output)+ self.W2(hidden_with_time_axis)))
        # FC = Fully connected (dense) layer   EO = Encoder output   H = hidden state    X = input to the decoder
        # score = FC(tanh(FC(EO) + FC(H)))


        # Calculate attention distribution
        """
        归一化score，得到attn_dist
        """
        attn_dist = tf.actitivation.softmax(score, axis=1)
        #reference:https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
        #attention weights = softmax(score, axis = 1) 因为softmax默认最后一行 axis=1改为第一行


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_,
                                                   weights=[embedding_matrix],
                                                   trainable=False)        
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # self.dropout = tf.keras.layers.Dropout(0.5)
        """
        定义最后的fc层，用于预测词的概率
        your code
        """
        attn_dist = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # print('x is ', x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output = self.dropout(output)
        out = self.fc(output)

        return x, out, state


# help(word2vec)