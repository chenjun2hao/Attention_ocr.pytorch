import lmdb
import six
from PIL import Image

env = lmdb.open('lmdb/test_lmdb',
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

label_fp = open('out/labels.txt', 'w')
with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'))
    #print nSamples
    for index in range(nSamples):
        image_key = 'image-%09d' % (index+1)
        label_key = 'label-%09d' % (index+1)
        imgbuf = txn.get(image_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf)
            savename = "out/%06d.png" % (index+1)
            img.save(savename)
            print("save %s" % savename)
        except IOError:
            print('Corrupted image for %d' % index)
        label = txn.get(label_key)
        print >> label_fp, label
label_fp.close() 
