with open('data/mnt/ramdisk/max/90kDICT32px/annotation_train.txt') as fp:
    lines = fp.readlines()

train_fp = open('data/train_list.txt', 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = 'data/mnt/ramdisk/max/90kDICT32px/%s' % imgpath
    output = ' '.join([imgpath, label])
    print >> train_fp, output

train_fp.close()


with open('data/mnt/ramdisk/max/90kDICT32px/annotation_test.txt') as fp:
    lines = fp.readlines()

test_fp = open('data/test_list.txt', 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = 'data/mnt/ramdisk/max/90kDICT32px/%s' % imgpath
    output = ' '.join([imgpath, label])
    print >> test_fp, output

test_fp.close()

with open('data/mnt/ramdisk/max/90kDICT32px/annotation_test.txt') as fp:
    lines = fp.readlines()

val_fp = open('data/val_list.txt', 'w')
for line in lines:
    imgpath = line.strip().split(' ')[0]
    label = imgpath.split('/')[-1].split('_')[1].lower()
    label = label + '$'
    label = ':'.join(label)
    imgpath = 'data/mnt/ramdisk/max/90kDICT32px/%s' % imgpath
    output = ' '.join([imgpath, label])
    print >> val_fp, output

val_fp.close()
