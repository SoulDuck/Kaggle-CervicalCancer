import numpy as np 
import matplotlib.pyplot as plt 
import PIL 
import glob
from PIL import Image
import os ,sys



class batch():
    def __init__(self ,folder_path , extension = '*.jpg'):
        self.folder_path = folder_path
        train_1_paths=glob.glob(folder_path+'train/Type_1/'+extension)
        train_2_paths=glob.glob(folder_path+'train/Type_2/'+extension)
        train_3_paths=glob.glob(folder_path+'train/Type_3/'+extension)
        add_1_paths=glob.glob(folder_path+'additional/Type_1/'+extension)
        add_2_paths=glob.glob(folder_path+'additional/Type_2/'+extension)
        add_3_paths=glob.glob(folder_path+'additional/Type_3/'+extension)

        print '# additional type1 :'+len(add_1_paths)
        print '# additional type2 :'+len(add_2_paths)
        print '# additional type3 :'+len(add_3_paths)
        print '# traintype1 :'+len(train_1_paths)
        print '# traintype2 :'+len(train_2_paths)
        print '# traintype3 :'+len(train_3_paths)

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
        for i,image in enumerate(images):
            imgs_np[i] = image
        return imgs_np
    def get_batch(self):
        type_1_imgs=self.makeBatch(train_1 , additional_1)
        type_2_imgs=self.makeBatch(train_2 , additional_2)
        type_3_imgs=self.makeBatch(train_3 , additional_3)
        return type1_imgs , type2_imgs , type3_imgs
        