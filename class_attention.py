# coding:utf-8
import utils
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
import models.crnn_lang as crnn

class attention_ocr():
    '''使用attention_ocr进行字符识别
    返回：
        ocr读数，置信度
    '''
    def __init__(self):
        encoder_path = './expr/attentioncnn/encoder_600.pth'
        decoder_path = './expr/attentioncnn/decoder_600.pth'
        self.alphabet = '0123456789'
        self.max_length = 7                          # 最长字符串的长度
        self.EOS_TOKEN = 1
        self.use_gpu = True
        self.max_width = 220
        self.converter = utils.strLabelConverterForAttention(self.alphabet)
        self.transform = transforms.ToTensor()

        nclass = len(self.alphabet) + 3
        encoder = crnn.CNN(32, 1, 256)          # 编码器
        decoder = crnn.decoder(256, nclass)  # seq to seq的解码器, nclass在decoder中还加了2

        if encoder_path and decoder_path:
            print('loading pretrained models ......')
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
        if torch.cuda.is_available() and self.use_gpu:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

    def constant_pad(self, img_crop):
        '''把图片等比例缩放到高度：32,再或resize填充到220宽度
        img_crop：
            cv2图片，rgb顺序
        返回：
            img tensor
        '''
        h, w, c = img_crop.shape
        ratio = h / 32
        new_w = int(w / ratio)
        new_img = cv2.resize(img_crop,(new_w, 32))
        container = np.ones((32, self.max_width, 3), dtype=np.uint8) * new_img[-3,-3,:]
        if new_w <= self.max_width:
            container[:,:new_w,:] = new_img
        elif new_w > self.max_width:
            container = cv2.resize(new_img, (self.max_width, 32))

        img = Image.fromarray(container.astype('uint8')).convert('L')
        img = self.transform(img)
        img.sub_(0.5).div_(0.5)
        if self.use_gpu:
            img = img.cuda()
        return img.unsqueeze(0)
    
    def predict(self, img_crop):
        '''attention ocr 做文字识别
        img_crop:
            cv2图片，rgb顺序
        返回：
            ocr读数，prob置信度
        '''
        img_tensor = self.constant_pad(img_crop)
        encoder_out = self.encoder(img_tensor)

        decoded_words = []
        prob = 1.0
        decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
        decoder_hidden = self.decoder.initHidden(1)
        if torch.cuda.is_available() and self.use_gpu:
            decoder_input = decoder_input.cuda()
            decoder_hidden = decoder_hidden.cuda()
        # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
        for di in range(self.max_length):  # 最大字符串的长度
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_out)
            probs = torch.exp(decoder_output)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            prob *= probs[:, ni]
            if ni == self.EOS_TOKEN:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.converter.decode(ni))

        words = ''.join(decoded_words)
        prob = prob.item()

        return words, prob

if __name__ == '__main__':
    path = './test_img/00027_299021_27.jpg'
    img = cv2.imread(path)
    attention = attention_ocr()
    res = attention.predict(img)
    print(res)