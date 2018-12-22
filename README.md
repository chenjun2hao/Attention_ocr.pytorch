attention-ocr.pytorch:Encoder+Decoder+attention model
======================================

This repository implements the the encoder and decoder model with attention model for OCR, and this repository is modified from https://github.com/meijieru/crnn.pytorch  
There are many attention ocr repository which is finished with tensorflow, but they don't give the **inference.py**, besides i'm not good at tensorflow, i can't finish the **inference.py** by myself

# requirements
```
pytorch 0.4.1
```

# Test
1. change the parameters of the **demo.py***
2.
```bash
python demo.py
```
3. results
```
>>>predict_str:87635                => prob:0.8684815168380737
```

# Train Your Owm Model
there are some details for attention
1. training and inferencing the width of image must be the same, in my project, i pad all the image's width to 220
2. **decoder(opt.nh, nclass, dropout_p=0.1, max_length=56)**, 'max_length' is the feature's width from encoder(change with the imgW)
3. for batch training,i pad the target label for the same length, and i encode the alphabet start from 3, 0 for SOS, 1 for EOS, 2 for $(means others)
4. the train_list.txt and test_list.txt are created as the follow form:
```
# path/to/image_name.jpg label
/media/chenjun/ed/18_MechanicalCrnn/data/mechanical/imgs/4667.jpg 99996
/media/chenjun/ed/18_MechanicalCrnn/data/mechanical/imgs/0985.jpg 81309
```

# Reference
1. [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
2. [Attention-OCR](https://github.com/da03/Attention-OCR)
3. [Seq2Seq-PyTorch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)

# TO DO
- [ ] change LSTM to Conv1D, it can greatly accelerate the inference
- [ ] to support images of different widths

# Other
I am now working in a company in chengdu, using deep learning to do image-related work. But the department is just established, no technical accumulation, the work is very difficult. So now I want to change a job. The place of work is either chengdu or chongqing. If there is a way, please help me push it internally. Thank you very much.  
本人现在在成都的一家公司，职位：图像识别算法工程。但是部门刚成立，招的都是应届生，没有技术积累。所以现在想换一份工作，做computer vision方向的，工作地点在成都或者重庆都行，有途径也请帮忙内推一下。本人练习方式：778961303@qq.com.非常感谢。
