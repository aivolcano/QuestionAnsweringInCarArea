from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() #删除空格
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model

    # 通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    #     global w2v

    w2v = Word2Vec(sentences=LineSentence(sentence_path), sg=1, size=256, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)
    #用LineSentence把一个txt文件转为所需要的格式 PathLineSentence把一个文件夹里所有text转为一句话一个列表。
    # w2v.save('word2vec.model')
    # loaded_model = Word2Vec.load('word2vec.model')
    # wv = w2v.wv
    # del w2v
    # wv.save('word_vector')

    def cal_similarity(self, test_word_id):
        sim_matrix = self.sess.run(self.similarity, feed_dict = {self.test})

    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))

# gensim.models说明
# sg = 1是skip - gram算法，对低频词敏感；默认sg = 0为CBOW算法。
#
# size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
#
# window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3 - b个词，后面看b个词（b在0 - 3之间随机）。
#
# min_count是对词进行过滤，频率小于min - count的单词则会被忽视，默认值为5。
#
# negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e - 3。
# hs = 1表示层级softmax将会被使用，默认hs = 0且negative不为0，则负采样将会被选择使用。
#
# workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。

