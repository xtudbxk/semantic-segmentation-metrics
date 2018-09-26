import os
import sys
import time
import math
import traceback
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from scipy import ndimage as nd
from .crf import crf_inference
from .dataset import dataset

def single_crf_metrics(params):
    img,label,featmap,crf_config,category_num,id_,output_dir = params
    m = metrics_np(n_class=category_num)
    crf_output = crf_inference(img,crf_config,category_num,featmap,use_log=True)
    crf_pred = np.argmax(crf_output,axis=2)

    if output_dir is not None:
        imgio.imsave(os.path.join(output_dir,"%s_img.png"%id_),img/256.0)
        imgio.imsave(os.path.join(output_dir,"%s_output.png"%id_),dataset.label2rgb(np.argmax(featmap,axis=2)))
        imgio.imsave(os.path.join(output_dir,"%s_label.png"%id_),dataset.label2rgb(label[:,:,0]))
        imgio.imsave(os.path.join(output_dir,"%s_pred.png"%id_),dataset.label2rgb(crf_pred))

    m.update(label,crf_pred)
    return m.hist

class Predict():
    def __init__(self,config):
        self.config = config
        self.crf_config = config.get("crf",None)
        self.category_num = self.config.get("category_num",21)
        self.input_size = self.config.get("input_size",(240,240)) # (w,h)
        if self.input_size is not None:
            self.h,self.w = self.input_size
        else:
            self.h,self.w = None,None
   
        assert "sess" in self.config, "no session in config while using existing net"
        self.sess = self.config["sess"]
        assert "net" in self.config, "no network in config while using existing net"
        self.net = self.config["net"]
        assert "data" in self.config, "no dataset in config while using existing net"
        self.data = self.config["data"]

    def metrics_predict_tf_with_crf(self,category="val",multiprocess_num=100,crf_config=None,scales=[1.0],fixed_input_size=None,output_dir=None,use_max=False):
        print("predict config:")
        print("category:%s,\n multiprocess_num:%d,\n crf_config:%s,\n scales:%s,\n fixed_input_size:%s,\n output_dir:%s,\n use_max:%s" % (category,multiprocess_num,str(crf_config),str(scales),str(fixed_input_size),str(output_dir),str(use_max)))


        pool = Pool(multiprocess_num)
        i = 0
        m = metrics_np(n_class=self.category_num)
        try:
            params = []
            while(True):
                label,img,id_ = self.sess.run([self.net["label"],self.net["input"],self.net["id"]])
                origin_h,origin_w = img.shape[1:3]
                origin_img = img
                if fixed_input_size is not None:
                    img = nd.zoom(img,[1.0,fixed_input_size[0]/img.shape[1],fixed_input_size[1]/img.shape[2],1.0],order=1)
                output = np.zeros([1,origin_h,origin_w,self.category_num])
                final_output = np.zeros([1,origin_h,origin_w,self.category_num])
                for scale in scales:
                    scale_1 = 1.0/scale
                    img_scale = nd.zoom(img,[1.0,scale,scale,1.0],order=1)
                    output_scale = self.sess.run(self.net["rescale_output"],feed_dict={self.net["input"]:img_scale})
                    output_scale = nd.zoom(output_scale,[1.0,origin_h/output_scale.shape[1],origin_w/output_scale.shape[2],1.0],order=0)
                    output_scale_h,output_scale_w = output_scale.shape[1:3]
                    output_h_ = min(origin_h,output_scale_h)
                    output_w_ = min(origin_w,output_scale_w)
                    final_output[:,:output_h_,:output_w_,:] = output_scale[:,:output_h_,:output_w_,:]
                    if use_max is True:
                        output = np.max(np.stack([output,final_output],axis=4),axis=4)
                    else:
                        output += final_output

                params.append((origin_img[0]+self.data.img_mean,label[0],output[0],crf_config,self.category_num,id_[0].decode(),output_dir))
                if i % multiprocess_num == multiprocess_num -1:
                    print("start %d ..." % i)
                    #print("params:%d" % len(params))
                    #print("params[0]:%d" % len(params[0]))
                    if len(params) > 0:
                        ret = pool.map(single_crf_metrics,params)
                        for hist in ret:
                            m.update_hist(hist)
                    params = []
                i += 1
        except tf.errors.OutOfRangeError:
            if len(params) > 0:
                ret = pool.map(single_crf_metrics,params)
                for hist in ret:
                    m.update_hist(hist)
            print("output of range")
            print("tf miou:%f" % m.get("miou"))
            print("all metrics:%s" % str(m.get_all()))
        except Exception as e:
            print("exception info:%s" % traceback.format_exc())
        finally:
            pool.close()
            pool.join()
            print("finally")

# Originally written by wkentaro for the numpy version
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
class metrics_np():
    def __init__(self,n_class=1,hist=None):
        if hist is None:
            self.hist = np.zeros((n_class,n_class))
        else:
            self.hist = hist
        self.n_class = n_class

    def _fast_hist(self,label_true,label_pred,n_class):
        mask = (label_true>=0) & (label_true<n_class) # to ignore void label
        self.hist = np.bincount( n_class * label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
        return self.hist

    def update(self,x,y):
        self.hist += self._fast_hist(x.flatten(),y.flatten(),self.n_class)

    def update_hist(self,hist):
        self.hist += hist

    def get(self,kind="miou"):
        if kind == "accu":
            return np.diag(self.hist).sum() / (self.hist.sum()+1e-3) # total pixel accuracy
        elif kind == "precision":
            return np.diag(self.hist) / (self.hist.sum(axis=0)+1e-3) 
        elif kind == "recall":
            return np.diag(self.hist) / (self.hist.sum(axis=1)+1e-3) 
        elif kind in ["freq","fiou","iou","miou"]:
            iou = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist)+1e-3)
            if kind == "iou": return iou
            miou = np.nanmean(iou)
            if kind == "miou": return miou

            freq = self.hist.sum(axis=1) / (self.hist.sum()+1e-3) # the frequency for each categorys
            if kind == "freq": return freq
            else: return (freq[freq>0]*iou[freq>0]).sum()
        elif kind in ["dice","mdice"]:
            dice = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)+1e-3)
            if kind == "dice": return dice
            else: return np.nanmean(dice)
        return None

    def get_all(self):
     metrics = {}
     metrics["accu"] = np.diag(self.hist).sum() / (self.hist.sum()+1e-3) # total pixel accuracy
     metrics["precision"] = np.diag(self.hist) / (self.hist.sum(axis=0)+1e-3) # pixel accuracys for each category, np.nan represent the corresponding category not exists
     metrics["recall"] = np.diag(self.hist) / (self.hist.sum(axis=1)+1e-3) # pixel accuracys for each category, np.nan represent the corresponding category not exists
     metrics["iou"] = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist)+1e-3)
     metrics["miou"] = np.nanmean(metrics["iou"])
     metrics["freq"] = self.hist.sum(axis=1) / (self.hist.sum()+1e-3) # the frequency for each categorys
     metrics["fiou"] = (metrics["freq"][metrics["freq"]>0]*metrics["iou"][metrics["freq"]>0]).sum()
     metrics["dices"] = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0)+1e-3)
     metrics["mdice"] = np.nanmean(metrics["dices"])
 
     return metrics

