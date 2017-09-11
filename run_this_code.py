#-*- coding:utf-8 -*-

from batch import tfrecord_pipeline,reconstruct_tfrecord_rawdata , get_shuffled_batch , get_iterator
import glob , os , sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def get_paths_from_text(text_locate):
    f=open(text_locate , 'r')
    lines=f.readlines()
    lines=map(lambda x: x.replace('\n' , '' ) , lines)

    return lines

paths=glob.glob('../fundus_data/cropped_original_fundus_300x300/cataract/*.png')

n=np.zeros([len(paths)])

batch =tfrecord_pipeline(paths , n)
print get_iterator('./sample.tfrecord')

#batch.make_tfrecord_rawdata('./sample.tfrecord')
#imgs, labs , names=reconstruct_tfrecord_rawdata('./sample.tfrecord')
imgs , labs  =get_shuffled_batch('./sample.tfrecord',1,(300,300))

print imgs
print np.shape(imgs)
print np.shape(labs)
coord = tf.train.Coordinator()
sess= tf.Session()
threads = tf.train.start_queue_runners(sess, coord)
for i in range(3):
    print i
    batch_xs,batch_ys=sess.run([imgs, labs])


coord.request_stop()
coord.join(threads)
#print np.shape(images)



#plt.imshow(imgs[0])
#plt.show()
"""
tfrecord_path ='./sample_images/sample_2.tfrecord'
images , labels  = batch.get_shuffled_batch(tfrecord_path,  3 , (224,224))
# 3은 batch_size 입니다 , (224,224)는 복원될 이미지 사이즈 입니다. 
# tf.bilinear라는 lib을 써서 작은 사진도 224,224로 복원됩니다. 
init_op=tf.group(tf.global_variables_initializer() ,  tf.local_variables_initializer())
sess= tf.Session()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess= sess, coord =coord)
for i in xrange(3):
    batch_xs , batch_ys=imgs,labs=sess.run([images , labels])
    #batch_xs ,batch_ys 을 이용하면 됩니다.
    print np.shape(batch_xs)
    print batch_ys
coord.request_stop()
coord.join(threads)

"""