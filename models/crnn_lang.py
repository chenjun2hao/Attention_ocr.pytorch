# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

GO = 0
EOS_TOKEN = 1              # 结束标志的标签

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
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.processed_batches = 0

    def forward(self, prev_hidden, feats, cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        self.processed_batches = self.processed_batches + 1

        if self.processed_batches % 10000 == 0:
            print('processed_batches = %d' %(self.processed_batches))

        alpha = F.softmax(emition) # nB * nT
        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC//感觉不应该sum，输出4×256
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha


class DecoderRNN(nn.Module):
    """
        采用RNN进行解码
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        return result


class Attentiondecoder(nn.Module):
    """
        采用attention注意力机制，进行解码
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Attentiondecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)         # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1) # 上一次的输出和隐藏状态求出权重
        attn_applied = torch.matmul(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute((1, 0, 2)))      # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        output = torch.cat((embedded, attn_applied.squeeze(1) ), 1)       # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)          # 最后输出一个概率
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result

    # def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128):
    #     super(Attention, self).__init__()
    #     self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings)
    #     self.input_size = input_size
    #     self.hidden_size = hidden_size
    #     self.generator = nn.Linear(hidden_size, num_classes)
    #     self.char_embeddings = Parameter(torch.randn(num_classes+1, num_embeddings))
    #     self.num_embeddings = num_embeddings
    #     self.processed_batches = 0

    # # targets is nT * nB
    # def forward(self, feats, text_length, text):
    #     # target_txt_decode
    #     targets =target_txt_decode(batch_size, text_length, text)

    #     output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
    #     hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
    #     max_locs = torch.zeros(num_steps, nB)
    #     max_vals = torch.zeros(num_steps, nB)
    #     for i in range(num_steps):
    #         cur_embeddings = self.char_embeddings.index_select(0, targets[i])
    #         hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings)
    #         output_hiddens[i] = hidden
    #         if self.processed_batches % 500 == 0:
    #             max_val, max_loc = alpha.data.max(1)
    #             max_locs[i] = max_loc.cpu()
    #             max_vals[i] = max_val.cpu()
    #     if self.processed_batches % 500 == 0:
    #         print('max_locs', list(max_locs[0:text_length.data[0],0]))
    #         print('max_vals', list(max_vals[0:text_length.data[0],0]))
    #     new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
    #     b = 0
    #     start = 0
    #     for length in text_length.data:
    #         new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
    #         start = start + length
    #         b = b + 1
    #     probs = self.generator(new_hiddens)
    #     return probs


def target_txt_decode(batch_size, text_length, text):
    '''
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
    return:
        targets: num_steps+1 * batch_size
    '''
    nB = batch_size      # batch

    # 将text分离出来
    num_steps = text_length.data.max()
    num_steps = int(num_steps.cpu().numpy())
    targets = torch.ones(nB, num_steps + 2) * 2                 # 用$符号填充较短的字符串, 在最开始加上GO,结束加上EOS_TOKEN
    targets = targets.long().cuda()        # 用
    start_id = 0
    for i in range(nB):
        targets[i][0] = GO    # 在开始的加上开始标签
        targets[i][1:text_length.data[i] + 1] = text.data[start_id:start_id+text_length.data[i]]       # 是否要加1
        targets[i][text_length.data[i] + 1] = EOS_TOKEN         # 加上结束标签
        start_id = start_id+text_length.data[i]                 # 拆分每个目标的target label，为：batch×最长字符的numel
    targets = Variable(targets.transpose(0, 1).contiguous())
    return targets
    

class CNN(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''
    def __init__(self, imgH, nc, nh):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)          # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）
        
        return encoder_outputs


class decoder(nn.Module):
    '''
        decoder from image features
    '''
    def __init__(self, nh=256, nclass=13, dropout_p=0.1, max_length=71):
        super(decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


        # target_variable = target_txt_decode(b, length, text)
        # decoder_input = torch.zeros(b).long().cuda()   # 初始化decoder的开始,从0开始输出
        # decoder_hidden = self.decoder.initHidden(b).cuda()
        # if self.teach_forcing:
        #     # 教师强制：将目标label作为下一个输入
        #     for di in range(target_variable.shape[0]):           # 最大字符串的长度
        #         decoder_output, decoder_hidden, decoder_attention = self.decoder(
        #             decoder_input, decoder_hidden, encoder_outputs)
        #         loss += criterion(decoder_output, target_variable[di])          # 每次预测一个字符
        #         decoder_input = target_variable[di]  # Teacher forcing/前一次的输出