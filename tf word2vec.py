
import tensorflow as tf

class word2vec():
    def __init__(self,
                 vocab_list = None,
                 embedding_size = 256,
                 win_len = 3,
                 learning_rate = 1,
                 num_sampled = 100):
        self.batch_size = None
        assert type(vocab_list) == list
        self.vocab_list= vocab_list
        self.vocab_size = vocab_list.__len__()
        self.win_len = win_len   #左边滑动多长 右边滑动多长
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate

        self.word2vec = {}
        for i in range(self.vocab_size):
            self.word2id[self.vocab_list[i]] = i

            self.train_words_num = 0
            self.train_sentence_num = 0
            self.train_times = 0

            self.word2id={}
            for i in range(self.vocab_size):
                self.word2id(self.vocab_list[i]) = i

            self.train_words_num = 0
            self.train_sentence_num = 0
            self.train_times = 0
        self.build_graph()

        def build_graph(self):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.train_inputs = tf.placeholder(tf.int32, shape = [self.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                self.embedding_dict = tf.Variable(tf.truncated_normal(shape=[vocab_size],embedding_size))
                self.nec_weight = tf.truncated_normal(shape=[vocab_size, embedding_size])
                self.bias = tf.Variable(tf.zeros(self.vocab_size))

                embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(weight=self.nec_weight,
                                   biases=self.bias,
                                   inputs=embed,
                                   labels=self.train_labels,
                                   num_sampled=self.num_sampled,#n个类别
                                   num_classes=self.vocab_size
                                   )
                )
                self.train_op=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss, global_gradients, aggregation_method, )
                self.test_word_id = tf.placeholder(tf.int32, shape=[None])

                vec_12_model = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict,1)))
                self.normed_embedding = self.embedding_dict/vec_12_model
                test_embed = tf.nn.embedding_lookup(self.embedding_dict,self.test_word_id)
                self.similarity(tf.matmul(test_embed,self.normed_embedding))

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()

    def train_by_sentence(self,input_sentence=[]):
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__):
                start = max(0, i-self.win_len)
                end = min(sent.__len__(), i+self.win_len)
                for index in range(start, end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent(index))
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        batch_inputs = np.array(batch_inputs, dtype = np.int32)
        batch_labels = np.array(batch_labels, dtype = np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels._len_(),1])

        feed_dict = {
            self.train_inputs:batch_inputs,
            self.train_labels:batch_labels
        }

        loss = self.sess.run(self.train_op,feed_dict=)

        self.train_words_num += batch_inputs._len_()
        self.train_sentence_num += input_sentence._len_()
        self.train_times += 1
