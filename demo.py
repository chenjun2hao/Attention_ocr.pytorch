# coding:utf-8

'''
March 2019 by Chen Jun
https://github.com/chenjun2hao/Attention_ocr.pytorch

'''

import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from utils import alphabet
import models.crnn_lang as crnn

use_gpu = True

encoder_path = './expr/attentioncnn/encoder_5.pth'
# decoder_path = './expr/attentioncnn/decoder_5.pth'
img_path = './test_img/20441531_4212871437.jpg'
max_length = 15                          # 最长字符串的长度
EOS_TOKEN = 1

nclass = len(alphabet) + 3
encoder = crnn.CNN(32, 1, 256)          # 编码器
# decoder = crnn.decoder(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
decoder = crnn.decoderV2(256, nclass)


if encoder_path and decoder_path:
    print('loading pretrained models ......')
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
if torch.cuda.is_available() and use_gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


converter = utils.strLabelConverterForAttention(alphabet)

transformer = dataset.resizeNormalize((280, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available() and use_gpu:
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

encoder.eval()
decoder.eval()
encoder_out = encoder(image)

decoded_words = []
prob = 1.0
decoder_attentions = torch.zeros(max_length, 71)
decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
decoder_hidden = decoder.initHidden(1)
if torch.cuda.is_available() and use_gpu:
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder_hidden.cuda()
loss = 0.0
# 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
for di in range(max_length):  # 最大字符串的长度
    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_out)
    probs = torch.exp(decoder_output)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)
    ni = topi.squeeze(1)
    decoder_input = ni
    prob *= probs[:, ni]
    if ni == EOS_TOKEN:
        # decoded_words.append('<EOS>')
        break
    else:
        decoded_words.append(converter.decode(ni))

words = ''.join(decoded_words)
prob = prob.item()
print('predict_str:%-20s => prob:%-20s' % (words, prob))
