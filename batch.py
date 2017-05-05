import numpy as np 
import matplotlib.pyplot as plt 
import PIL 
import glob
from PIL import Image
import os ,sys



class batch():
    def __init__(self ,folder_path , extension = '*.jpg'):
        self.folder_path = folder_path
        self.train_1_paths=glob.glob(folder_path+'train/Type_1/'+extension)
        self.train_2_paths=glob.glob(folder_path+'train/Type_2/'+extension)
        self.train_3_paths=glob.glob(folder_path+'train/Type_3/'+extension)
        self.add_1_paths=glob.glob(folder_path+'additional/Type_1/'+extension)
        self.add_2_paths=glob.glob(folder_path+'additional/Type_2/'+extension)
        self.add_3_paths=glob.glob(folder_path+'additional/Type_3/'+extension)

        print '# additional type1 :'+str(len(self.add_1_paths))
        print '# additional type2 :'+str(len(self.add_2_paths))
        print '# additional type3 :'+str(len(self.add_3_paths))
        print '# traintype1 :'+str(len(self.train_1_paths))
        print '# traintype2 :'+str(len(self.train_2_paths))
        print '# traintype3 :'+str(len(self.train_3_paths))

        self.training_type1_paths=[]
        self.training_type2_paths=[]
        self.training_type3_paths=[]
        self.test_type1_paths=[]
        self.test_type2_paths=[]
        self.test_type3_paths=[]

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
        for i,image in enume0rate(images):
            imgs_np[i] = image
        return imgs_np
    def get_batch(self):
        type_1_imgs=self.makeBatch(train_1 , additional_1)
        type_2_imgs=self.makeBatch(train_2 , additional_2)
        type_3_imgs=self.makeBatch(train_3 , additional_3)
        return type1_imgs , type2_imgs , type3_imgs
        