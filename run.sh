#CUDA_VISIBLE_DEVICES=0 nohup python main.py --experiment expr_basic --trainlist data/train_list.txt --vallist data/val_list.txt --cuda --adam --lr=0.001 > log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --lang --experiment expr_basic_lang --trainlist data/train_list.txt --vallist data/val_list.txt --cuda --adam --lr=0.001 > log_lang.txt &
