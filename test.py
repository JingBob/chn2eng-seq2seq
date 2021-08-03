import torch
from chinese_datapre import prepareData
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from evaluate import evaluateRandomly, evaluate, showAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'chin', True)
    hidden_size = 256  # 隐藏层维度设置为256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    # 恢复网络
    encoder1.load_state_dict(torch.load('models/encoder.pkl'))
    attn_decoder1.load_state_dict(torch.load('models/decoder.pkl'))

    # 随机从数据集选几个句子翻译下
    evaluateRandomly(input_lang, output_lang, pairs, encoder1, attn_decoder1)

    # 输入句子测试下，句子的单词必须是数据集里有的，否则报错
    evaluateAndShowAttention("我喜欢你")
