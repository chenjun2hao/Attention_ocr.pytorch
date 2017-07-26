### Train on vgg recogniton txt
- download mjsynth.tar.gz and unzip to current folder
- copy annotation_train.txt annotation_test.txt annotation_val.txt to current
- correct path info
- create imagelist: cat annotation_train.imgs | awk -F / '{print $NF}' | awk -F _ '{print $2}' | tr [:upper:] [:lower:]
- python create_dataset.py
