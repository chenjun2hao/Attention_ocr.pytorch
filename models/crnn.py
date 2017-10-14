import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, prev_hidden, feats):
        nT = feats.size(0)
        nB = feats.size(1)
        assert(nB == 1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        alpha = F.softmax(emition) # nB * nT
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)

    def forward(self, feats, text_length):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_labels = text_length.data.sum()

        output_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
        k = 0
        for j in range(nB):
            sub_feats = feats[:,j,:].contiguous().view(nT,1,nC) #feats.index_select(1, Variable(torch.LongTensor([j]).type_as(feats.data)))
            sub_hidden = Variable(torch.zeros(1,hidden_size).type_as(feats.data))
            for i in range(text_length.data[j]):
                sub_hidden, sub_alpha = self.attention_cell(sub_hidden, sub_feats)
                output_hiddens[k] = sub_hidden.view(-1)
                k = k + 1
        probs = self.generator(output_hiddens)
        return probs

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        #self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))
        self.attention = Attention(nh, nh/2, nclass)

    def forward(self, input, length):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)
        output = self.attention(rnn, length)

        return output
