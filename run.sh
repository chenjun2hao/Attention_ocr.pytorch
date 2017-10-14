# the format or list file
# imagepath label
# /ab/cd/image.jpg a:b:c:d 

#python main.py --trainlist train_list2.txt --vallist test_list2.txt --cuda --adam --lr=0.001



#python main_for_music.py --trainlist data/train_list2.txt --vallist data/test_list2.txt --cuda --adam --lr=0.001
# loss ~ 2



nohup python main_for_music.py --trainlist data/train_list3.txt --vallist data/test_list3.txt --cuda --adam --lr=0.001 > log3.txt &
# loss ~ 2
