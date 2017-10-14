# the format or list file
# imagepath label
# /ab/cd/image.jpg a:b:c:d 

python main.py --trainlist data/train_list.txt --vallist data/test_list.txt --cuda --adam --lr=0.001
