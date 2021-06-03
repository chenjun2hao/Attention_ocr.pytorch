#!/usr/bin/python
# encoding: utf-8
#  -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from PIL import Image, ImageFilter
import math
import random
import numpy as np
import cv2

with open('./data/devanagari-charset.txt',encoding="utf-8") as f:
    data = f.readlines()
    alphabet = [" "]
    alphabet += [x.rstrip() for x in data]
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
        self.dict['SOS'] = 0  # start
        self.dict['EOS'] = 1  # end
        self.dict['$'] = 2  # blank identifier
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i + 3                     # Encode from 3

    def encode(self, text):
        """Encode and align target_label
        Add GO to the beginning of each string of target txt, and EOS at the end, and use the longest string for alignment
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.dict[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]  # encoding

            max_length = max([len(x) for x in text])  # align
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2  # use ‘blank’ for padding
            for i in range(nb):
                targets[i][0] = 0  # start
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
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


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
    u: Mean value of Gaussian distribution
    sigma: variance
    nums: Randomly add several light points to a picture
    prob: probability of using halo
    '''

    def __init__(self, nums, u=0, sigma=0.2, prob=0.5):
        self.u = u  # mean value μ
        self.sig = math.sqrt(sigma)  # standard deviation δ
        self.nums = nums
        self.prob = prob

    def create_kernel(self, maxh=32, maxw=50):
        height_scope = [10, maxh]  # Height range randomly generated Gaussian
        weight_scope = [20, maxw]  # width range

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
            Gauss_map = self.create_kernel(32,60)  # Initialize a Gaussian kernel, 32 is the maximum value in the height direction, and 60 is the w direction
            img1 = np.asarray(img)
            img1.flags.writeable = True  # Change the array to read-write mode
            nums = random.randint(1, self.nums)  # Randomly generate nums light points
            img1 = img1.astype(np.float)
            # print(nums)
            for i in range(nums):
                img_h, img_w = img1.shape
                pointx = random.randint(0, img_h - 10)  # find a random point in the original image
                pointy = random.randint(0, img_w - 10)

                h, w = Gauss_map.shape  # Determine whether the limit is exceeded
                endx = pointx + h
                endy = pointy + w

                if pointx + h > img_h:
                    endx = img_h
                    Gauss_map = Gauss_map[1:img_h - pointx + 1, :]
                if img_w < pointy + w:
                    endy = img_w
                    Gauss_map = Gauss_map[:, 1:img_w - pointy + 1]

                # Plus uneven lighting
                img1[pointx:endx, pointy:endy] = img1[pointx:endx, pointy:endy] + Gauss_map * 255.0
            img1[img1 > 255.0] = 255.0  # limit, otherwise uint8 will start counting from 0
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
            adjust = random.choice([0.5, 0.7, 0.9, 1.2, 1.5, 1.7])  # choose one at random
            # adjust = random.choice([1.2, 1.5, 1.7, 2.0]) # Choose one at random
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(np.uint8(rgb)).convert('L')


class randapply(object):
    """Randomly decide whether to apply halo, blur, or both

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


def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
