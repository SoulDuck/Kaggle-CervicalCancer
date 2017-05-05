import numpy as np 
import matplotlib.pyplot as plt 
import PIL 
import glob
from PIL import Image
import os ,sys



class batch():
    def __init__(self ,folder_path):
        self.folder_path = folder_path
        train_1_paths=crawl_folder(folder_path+'train/Type_1/')
        train_2_paths=crawl_folder(folder_path+'train/Type_2/')
        train_3_paths=crawl_folder(folder_path+'train/Type_3/')
        add_1_paths=crawl_folder(folder_path+'additional/Type_1/')
        add_2_paths=crawl_folder(folder_path+'additional/Type_2/')
        add_3_paths=crawl_folder(folder_path+'additional/Type_3/')

        print len(add_1_paths)
        print len(add_2_paths)
        print len(add_3_paths)
        print len(train_1_paths)
        print len(train_2_paths)
        print len(train_3_paths)

        training_type1_paths=[]
        training_type2_paths=[]
        training_type3_paths=[]
        test_type1_paths=[]
        test_type2_paths=[]
        test_type3_paths=[]

    def makeBatch(self , *args):
        imgs=[]
        for thing in args:
            paths=list(thing) 
            for path in paths:
                try:
                    img=Image.open(path)
                    imgs.append(img)
                    img=resize_image(img)
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
        for i,image in enumerate(images):
            imgs_np[i] = image
        return imgs_np
    def get_batch(self):
        type_1_imgs=makeBatch(train_1 , additional_1)
        type_2_imgs=makeBatch(train_2 , additional_2)
        type_3_imgs=makeBatch(train_3 , additional_3)
        return type1_imgs , type2_imgs , type3_imgs
        