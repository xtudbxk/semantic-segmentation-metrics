import os
import sys
import math
import numpy as np
import tensorflow as tf
import skimage.color as imgco

class dataset():
    def __init__(self,config={}):
        self.config = config
        if self.config["input_size"] is not None:
            self.w,self.h = self.config.get("input_size",(240,240))
        else:
            self.w,self.h = None,None
        self.categorys = self.config.get("categorys",["train","val"])
        assert len(self.categorys) > 0, "no enough categorys in dataset"
        self.main_path = self.config.get("main_path",os.path.join("pascal","VOCdevkit","VOC2012"))
        #self.main_path = self.config.get("main_path",os.path.join("pascal"))
        self.ignore_label = self.config.get("ignore_label",255)
        self.category_num = self.config.get("category_num",21)
        self.default_category = self.config.get("default_category",self.categorys[0])
        self.img_mean = np.ones((1,1,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434

        self.data_f,self.data_len = self.get_data_f()

    def get_data_f(self):
        data_f = {}
        data_len = {}
        for category in self.categorys:
            data_f[category] = {"img":[],"label":[],"id":[]}
            data_len[category] = 0
        for one in self.categorys:
            with open(os.path.join("pascal","txt","%s.txt" % one),"r") as f:
                for line in f.readlines():
                    line = line.strip("\n") # the line is like "2007_000738"
                    #line = "2007_000332"
                    data_f[one]["id"].append(line)
                    data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % line))
                    data_f[one]["label"].append(os.path.join(self.main_path,"SegmentationClassAug","%s.png" % line))
                if "length" in self.config:
                    length = self.config["length"]
                    data_f[one]["id"] = data_f[one]["id"][:length]
                    data_f[one]["img"] = data_f[one]["img"][:length]
                    data_f[one]["label"] = data_f[one]["label"][:length]
            data_len[one] = len(data_f[one]["label"])

        print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,batch_size=None,epoches=-1):
        if category is None: category = self.default_category
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
        if self.h is None: assert batch_size == 1,"the input size is None, so the batch size must be 1"
        dataset = tf.data.Dataset.from_tensor_slices({
            "id":self.data_f[category]["id"],
            "img_f":self.data_f[category]["img"],
            "label_f":self.data_f[category]["label"]
            })
        def m(x):
            id = x["id"]
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img = tf.expand_dims(img,axis=0)
            label_f = x["label_f"]
            label_raw = tf.read_file(label_f)
            label = tf.image.decode_image(label_raw)
            label = tf.expand_dims(label,axis=0)
            if category == "train":
                img,label = self.image_preprocess(img,label,random_scale=True,flip=True,rotate=False)
                #img,label = self.image_preprocess(img,label,random_scale=False,flip=False,rotate=False)
            else:
                img,label = self.image_preprocess(img,label,random_scale=False,flip=False)

            #img,label = self.image_preprocess(img,label,random_scale=True,flip=True,rotate=False)
            if self.h is not None:
                img = tf.reshape(img,[self.h,self.w,3])
                label = tf.reshape(label,[self.h,self.w,1])
            label = tf.cast(label,tf.int32)

            return img,label,id

        dataset = dataset.repeat(epoches)
        dataset = dataset.shuffle(self.data_len[category])
        dataset = dataset.map(m)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        img,label,id = iterator.get_next()
            
        return img,label,id,iterator

    def image_preprocess(self,img,label,random_scale=False,flip=False,rotate=False,crop_and_pad=True):
        # input img and label shape [None, h, w, c]
        # NOTE random_scale and crop_and_pad is not compatiable
        label -= self.ignore_label
        if random_scale is True:
            scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.to_float(tf.shape(img)[1])* scale)
            w_new = tf.to_int32(tf.to_float(tf.shape(img)[2])* scale)
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
            img = tf.image.resize_bilinear(img, new_shape)
            img = tf.squeeze(img, squeeze_dims=[0])
            label = tf.image.resize_nearest_neighbor(label, new_shape)
            label = tf.squeeze(label, squeeze_dims=[0])
        else:
            img = tf.squeeze(img, squeeze_dims=[0])
            label = tf.squeeze(label, squeeze_dims=[0])
        label += self.ignore_label

        if self.h is not None:
            if crop_and_pad is True:
                img = tf.image.resize_image_with_crop_or_pad(img, self.h, self.w)
                label = tf.image.resize_image_with_crop_or_pad(label, self.h, self.w)
            else:
                img = tf.expand_dims(img,axis=0)
                img = tf.image.resize_bilinear(img,(self.h,self.w))
                img = tf.squeeze(img,axis=0)
                label = tf.expand_dims(label,axis=0)
                label = tf.image.resize_nearest_neighbor(label,(self.h,self.w))
                label = tf.squeeze(label,axis=0)

        r,g,b = tf.split(axis=2,num_or_size_splits=3,value=img)
        img = tf.cast(tf.concat([b,g,r],2),dtype=tf.float32)
        img -= self.img_mean

        if flip is True:
            img,label = self.image_flip(img,label)

        if rotate is True:
            img,label = self.image_rotate(img,label,minangle=-math.pi/18.0, maxangle = math.pi/18.0)

        return img, label

    def image_flip(self,img,label,left_right=True, up_down=False, random_s=0.5):
        if left_right is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            img = tf.cond(r < random_s, lambda:tf.image.flip_left_right(img),lambda:img)
            label = tf.cond(r < random_s, lambda:tf.image.flip_left_right(label),lambda:label)
        if up_down is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            img = tf.cond(r < random_s, lambda:tf.image.flip_left_right(img),lambda:img)
            label = tf.cond(r < random_s, lambda:tf.image.flip_left_right(label),lambda:label)

        return img, label

    def image_rotate(self,img,label,minangle=0,maxangle=0.314):
        angle = tf.random_uniform([1],minval=minangle,maxval=maxangle)
        angle = tf.squeeze(angle,[0])
        img = tf.contrib.image.rotate(img, angle)
        label = tf.contrib.image.rotate(label,angle)
        return img, label

    label2rgb_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128,0 ), (128, 128, 0),
                      (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                      (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                      (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                      (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                      (0, 64, 128)]) # using palette for pascal voc
    @staticmethod
    def label2rgb(label,colors=[],ignore_label=128,ignore_color=(255,255,255)):
        if len(colors) <= 0: 
            index = np.unique(label)
            #print("before index:%s" % str(index))
            index = index[index<21]
            #print("after index:%s" % str(index))
            colors = dataset.label2rgb_colors[index]
        label = imgco.label2rgb(label,colors=colors,bg_label=ignore_label,bg_color=ignore_color)
        return label.astype(np.uint8)
