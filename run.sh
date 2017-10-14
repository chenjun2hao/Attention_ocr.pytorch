#python main.py --trainroot ../PyTorch/crnn/tool/data/train_lmdb/ --valroot ../PyTorch/crnn/tool/data/test_lmdb/ --cuda --adam --lr=0.001
python main.py --trainlist train_list.txt --valroot ../PyTorch/crnn/tool/data/test_lmdb/ --cuda --adam --lr=0.001 # train_list could be annotation_train.txt
