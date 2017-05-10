import numpy as np 
import matplotlib.pyplot as plt 
import PIL 
import glob
from PIL import Image
import os ,sys
import tensorflow as tf


class batch():
    def __init__(self ,src_folder_paths  , labels, extension = '*.jpg' ):
        self.src_folder_paths = src_folder_paths
        self.src_folder_name=[]
        self.n_src_folder_files=[]
        self.labels= labels
        self.all_files_paths=[]
        self.all_files_labels=[]
        paths_labels=zip(self.src_folder_paths  , labels)
        for  i, (src_folder_path , label) in enumerate(paths_labels):
            paths=glob.glob(src_folder_path+extension)
            for path in paths:
                self.all_files_paths.append(path)
                self.all_files_labels.append(label)
            print '#'+src_folder_path+':'+str(len(paths))
            self.n_src_folder_files.append(len(paths)) 
        print 'the number of total files : ',len(self.all_files_paths) 
        print 'the number of label files : :',len(self.all_files_labels)

    def makeBatch(self , *args):
        imgs=[]
        for thing in args:
            paths=list(thing) 
            n = len(paths)
            for ind,path in enumerate(paths):
                msg = '\r-Progress : {0}'.format(str(ind) +'/'+str(n))
                sys.stdout.write(msg)
                sys.stdout.flush()

                sys.stdout.write(msg)
                sys.stdout.flush()
                try:
                    img=Image.open(path)
                    imgs.append(img)
                    img=self.resize_image(img)
                except IOError as ioe:
                    f=open('Error_log.txt' , 'a')
                    f.write(str(ioe))
                    f.write(path)
                    print path
                    continue

        return imgs

    def resize_image(self , img , size =(224,224)):
        img=img.resize(size , Image.ANTIALIAS)
        return img

    def Images2numpy(self , images , img_size=(224,224) , color_ch =3 ):
        n=len(images)
        imgs_np=np.zeros(n,img_size[0] , img_size[1] , color_ch )
        for i,image in enume0rate(images):
            imgs_np[i] = image
        return imgs_np
    def get_batch(self):
        type_1_imgs=self.makeBatch(self.train_1_paths, self.add_1_paths)
        type_2_imgs=self.makeBatch(self.train_2_paths, self.add_2_paths)
        type_3_imgs=self.makeBatch(self.train_3_paths, self.add_3_paths)
        return type1_imgs , type2_imgs , type3_imgs
    def get_name(self , path):
        name =path.split('/')[-1].split('.')[0]
        return name 
    def make_tfrecord_rawdata(self , tfrecord_path):        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
        def _int64_feature(value):
            return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        all_paths_labels=zip(self.all_files_paths , self.all_files_labels)
        error_file_paths=[]
        for ind, (path , label) in enumerate(all_paths_labels):
            try:
                msg = '\r-Progress : {0}'.format(str(ind) +'/'+str(len(all_paths_labels)))
                sys.stdout.write(msg)
                sys.stdout.flush()
                
                np_img=np.asarray(Image.open(path))
                height = np_img.shape[0]
                width = np_img.shape[1]
                raw_img = np_img.tostring()
                
                example = tf.train.Example(features = tf.train.Features(feature = {
                            'height': _int64_feature(height),
                            'width' : _int64_feature(width),
                            'raw_image' : _bytes_feature(raw_img),
                            'label' : _int64_feature(label)}))
                writer.write(example.SerializeToString())
            except IndexError as ie :
                print path
                continue
            except IOError as ioe:
                print path
                continue
            except Exception as e:
                print path
                print str(e)
                continue
        writer.close()
    def reconstruct_tfrecord_rawdata(self , tfrecord_path):
       
        if os.path.exists(tfrecord_path):
            print str(tfrecord_path) + 'already exist!'
            return 
        
        
        print 'now Reconstruct Image Data please wait a second'
        reconstruct_image=[]
        #caution record_iter is generator 
        
        record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)
        n=len(list(record_iter))
        record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)

        print 'The Number of Data :' , n
        ret_img_list=[]
        for i, str_record in enumerate(record_iter):
            msg = '\r -progress {0}/{1}'.format(i,n)
            sys.stdout.write(msg)
            sys.stdout.flush()

            example = tf.train.Example()
            example.ParseFromString(str_record)

            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            image = np.fromstring(raw_image , dtype = np.uint8)
            image = image.reshape((height , width , -1))
            ret_img_list.append(image)
            ret_lab_list.append(label)
        ret_img=np.asarray(ret_img_list)
        ret_lab=np.asarray(ret_lab_list)
        return ret_img , ret_lab
    def get_shuffled_batch(self , tfrecord_path , batch_size , resize ,  ):
        resize_height , resize_width  = resize
        filename_queue = tf.train.string_input_producer([tfrecord_path] , num_epochs=10)
        reader = tf.TFRecordReader()
        _ , serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'raw_image': tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([] , tf.int64)
            })
        image = tf.decode_raw(features['raw_image'], tf.uint8)
        height= tf.cast(features['height'] , tf.int32)
        width = tf.cast(features['width'] , tf.int32)
        label = tf.cast(features['label'] , tf.int32)
        image_shape = tf.pack([height , width , 3 ]) 
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image=tf.reshape(image ,  image_shape)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
        images  , labels= tf.train.shuffle_batch([image , label] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
        return images  , labels
