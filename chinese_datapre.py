from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import jieba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载的数据集包含一些其他信息，我们只保留前面的中文-英文对
# lines = open('data/%s-%s.txt' % ("eng", "chn"), encoding='utf-8').read().strip().split('\n')
# pairs = [[s for s in l.split('\t')] for l in lines]
#
# with open("data/eng-chin.txt", "w", encoding='utf-8') as f:
#     for i in pairs:
#         f.write(i[0]+'\t'+i[1]+'\n')


# 需要每个单词的唯一索引用作以后网络的输入和目标。
SOS_token = 0
EOS_token = 1
# 规定句子的最大长度
MAX_LENGTH = 30

# 用一个名为Lang的辅助类
# 包含 word → index ( word2index) 和 index → word ( index2word) 字典，以及每个单词的计数word2count，
# 稍后将用于替换稀有单词。
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # 如果是中文，不能用空格分词了，用结巴分词
        if self.name == "chin":
            words = jieba.cut(sentence)
        else:
            words = sentence.split(' ')
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 这些文件都是 Unicode 格式的，为了简化，我们将把 Unicode 字符转换为 ASCII，使所有内容都小写，并修剪大部分标点符号。
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 统一小写、修剪和删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


'''
为了读取数据文件，我们将文件分割成行，然后将行分割成对。
文件都是英语→其他语言，所以如果我们想从其他语言→英语翻译，我添加了reverse 标志来反转对。
'''
def readLangs(language1, language2, reverse=False):
    print("Reading lines...")

    # 分行
    lines = open('data/%s-%s.txt' % (language1, language2), encoding='utf-8'). \
        read().strip().split('\n')

    # 将每行分割成语言对，并正则化，这里用’\t‘即回车分割
    pairs=[]
    for l in lines:
        ss = l.split('\t')
        ss[0] = normalizeString(ss[0])
        ss[1] = unicodeToAscii(ss[1].lower().strip())
        pairs.append(ss)
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 反转语言对
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(language2)
        output_lang = Lang(language1)
    else:
        input_lang = Lang(language1)
        output_lang = Lang(language2)

    return input_lang, output_lang, pairs


'''
由于有很多例句，我们想快速训练一些东西，我们将把数据集修剪成只相对较短和简单的句子。
可以自行选择要不要过滤，如过滤则只保留开头是以下的句子
'''

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(list(jieba.cut(p[0]))) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH  # and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


'''
准备数据的完整过程是：
1.读取文本文件并拆分成行，将行拆分成对
2.标准化文本，按长度和内容过滤
3.从成对的句子制作单词列表
'''
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs




