#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from PIL import Image, ImageFilter
import math
import random
import numpy as np
import cv2

with open('./data/char_std_5990.txt') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)


class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS'] = 0       # 开始
        self.dict['EOS'] = 1       # 结束
        self.dict['$'] = 2         # blank标识符
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i + 3                     # 从3开始编码

    def encode(self, text):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.dict[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]           # 编码

            max_length = max([len(x) for x in text])        # 对齐
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2              # use ‘blank’ for pading
            for i in range(nb):
                targets[i][0] = 0                           # 开始
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = 1
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

class strLabelConverterForCTC(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self.sep = sep
        self.alphabet = alphabet.split(sep)
        self.alphabet.append('-')  # for `-1` index

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = text.split(self.sep)
            text = [self.dict[item] for item in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s.split(self.sep)) for s in text]
            text = self.sep.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


class halo():
    '''
    u:高斯分布的均值
    sigma:方差
    nums:在一张图片中随机添加几个光点
    prob:使用halo的概率
    '''

    def __init__(self, nums, u=0, sigma=0.2, prob=0.5):
        self.u = u  # 均值μ
        self.sig = math.sqrt(sigma)  # 标准差δ
        self.nums = nums
        self.prob = prob

    def create_kernel(self, maxh=32, maxw=50):
        height_scope = [10, maxh]  # 高度范围     随机生成高斯
        weight_scope = [20, maxw]  # 宽度范围

        x = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*height_scope))
        y = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*weight_scope))
        Gauss_map = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Gauss_map[i, j] = np.exp(-((x[i] - self.u) ** 2 + (y[j] - self.u) ** 2) / (2 * self.sig ** 2)) / (
                        math.sqrt(2 * math.pi) * self.sig)

        return Gauss_map

    def __call__(self, img):
        if random.random() < self.prob:
            Gauss_map = self.create_kernel(32, 60)  # 初始化一个高斯核,32为高度方向的最大值，60为w方向
            img1 = np.asarray(img)
            img1.flags.writeable = True  # 将数组改为读写模式
            nums = random.randint(1, self.nums)  # 随机生成nums个光点
            img1 = img1.astype(np.float)
            # print(nums)
            for i in range(nums):
                img_h, img_w = img1.shape
                pointx = random.randint(0, img_h - 10)  # 在原图中随机找一个点
                pointy = random.randint(0, img_w - 10)

                h, w = Gauss_map.shape  # 判断是否超限
                endx = pointx + h
                endy = pointy + w

                if pointx + h > img_h:
                    endx = img_h
                    Gauss_map = Gauss_map[1:img_h - pointx + 1, :]
                if img_w < pointy + w:
                    endy = img_w
                    Gauss_map = Gauss_map[:, 1:img_w - pointy + 1]

                # 加上不均匀光照
                img1[pointx:endx, pointy:endy] = img1[pointx:endx, pointy:endy] + Gauss_map * 255.0
            img1[img1 > 255.0] = 255.0  # 进行限幅，不然uint8会从0开始重新计数
            img = img1
        return Image.fromarray(np.uint8(img))


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


class GBlur(object):
    def __init__(self, radius=2, prob=0.5):
        radius = random.randint(0, radius)
        self.blur = MyGaussianBlur(radius=radius)
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = img.filter(self.blur)
        return img


class RandomBrightness(object):
    """随机改变亮度
        pil:pil格式的图片
    """

    def __init__(self, prob=1.5):
        self.prob = prob

    def __call__(self, pil):
        rgb = np.asarray(pil)
        if random.random() < self.prob:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 0.7, 0.9, 1.2, 1.5, 1.7])  # 随机选择一个
            # adjust = random.choice([1.2, 1.5, 1.7, 2.0])      # 随机选择一个
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(np.uint8(rgb)).convert('L')


class randapply(object):
    """随机决定是否应用光晕、模糊或者二者都用

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string