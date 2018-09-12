import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from pythonlib.crf import crf_inference
from pythonlib.dataset import dataset
from pythonlib.predict import Predict

class Test():
    def __init__(self,config):
        self.config = config
        if self.config["input_size"] is not None:
            self.h,self.w = self.config.get("input_size",(25,25))
        else:
            self.h,self.w = None,None
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.net = {}
        self.weights = {}
        self.min_prob = 0.0001
        self.stride = {}
        self.stride["input"] = 1
        self.trainable_list = []

    def build(self,net_input,net_label):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = net_input
                self.net["label"] = net_label # [None, self.h,self.w,1], int32
                self.net["drop_prob"] = tf.Variable(1.0)

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("vgg") as scope:
            # build block
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
            fc1 = self.build_fc(block,["fc6_1","relu6_1","drop6_1","fc7_1","relu7_1","drop7_1","fc8_1"], dilate_rate=6)
            fc2 = self.build_fc(block,["fc6_2","relu6_2","drop6_2","fc7_2","relu7_2","drop7_2","fc8_2"], dilate_rate=12)
            fc3 = self.build_fc(block,["fc6_3","relu6_3","drop6_3","fc7_3","relu7_3","drop7_3","fc8_3"], dilate_rate=18)
            fc4 = self.build_fc(block,["fc6_4","relu6_4","drop6_4","fc7_4","relu7_4","drop7_4","fc8_4"], dilate_rate=24)
            #self.net["fc8"] = (self.net[fc1]+self.net[fc2]+self.net[fc3]+self.net[fc4])/4.0
            self.net["fc8"] = self.net[fc1]+self.net[fc2]+self.net[fc3]+self.net[fc4]

            return self.net["fc8"] # note that the output is a log-number

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            elif layer.startswith("pool5a"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.avg_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                    last_layer = layer
            elif layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists,dilate_rate):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if layer.startswith("fc6"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=dilate_rate, padding="SAME", name="conv")

                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],self.net["drop_prob"])
                    last_layer = layer

        return last_layer

    def get_weights_and_bias(self,layer,shape=None):
        print("layer: %s" % layer)
        if layer in self.weights:
            return self.weights[layer]
        if shape is not None:
            pass
        elif layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[layer]
            if shape[3] > 512: shape[3] = 512
        elif layer.startswith("fc"):
            if layer.startswith("fc6"):
                shape = [3,3,512,1024]
            if layer.startswith("fc7"):
                shape = [1,1,1024,1024]
            if layer.startswith("fc8"): 
                shape = [1,1,1024,self.category_num]
        init = tf.random_normal_initializer(stddev=0.01)
        weights = tf.get_variable(name="%s_weights" % layer,initializer=init, shape = shape)
        init = tf.constant_initializer(0)
        bias = tf.get_variable(name="%s_bias" % layer,initializer=init, shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)

        return weights,bias

    def pred(self):
        if self.h is not None:
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],(self.h,self.w))
        else:
            label_size = tf.py_func(lambda x:x.shape[1:3],[self.net["input"]],[tf.int64,tf.int64])
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],[tf.cast(label_size[0],tf.int32),tf.cast(label_size[1],tf.int32)])
            
        self.net["pred"] = tf.argmax(self.net["rescale_output"],axis=3)

    def remove_ignore_label(self,gt,output=None,pred=None): 
        ''' 
        gt: not one-hot 
        output: a distriution of all labels, and is scaled to macth the size of gt
        NOTE the result is a flatted tensor
        and all label which is bigger that or equal to self.category_num is void label
        '''
        gt = tf.reshape(gt,shape=[-1])
        indices = tf.squeeze(tf.where(tf.less(gt,self.category_num)),axis=1)
        gt = tf.gather(gt,indices)
        if output is not None:
            output = tf.reshape(output, shape=[-1,self.category_num])
            output = tf.gather(output,indices)
            return gt,output
        elif pred is not None:
            pred = tf.reshape(pred, shape=[-1])
            pred = tf.gather(pred,indices)
            return gt,pred

    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)


    def predict(self):
        self.sess = tf.Session()

        data = dataset({"input_size":None,"categorys":["val"]}) # this is not same with self.data, note the input_size must be None
        data_x,data_y,_,iterator = data.next_batch(category="val",epoches=1)
        self.build(net_input=data_x,net_label=data_y)

        #crf_config = None
        config = {"input_size":None,"sess":self.sess,"net":self.net,"data":data}
        p = Predict(config)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(iterator.initializer)

        self.saver = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        if self.config.get("model_path",False) is not False:
            print("start to load model: %s" % self.config.get("model_path"))
            self.restore_from_model(self.saver,self.config.get("model_path"),checkpoint=False)
            print("model loaded ...")

        crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":80,"bi_srgb":13,"bi_compat":10,"iterations":5} # for test
        start_time = time.time()
        #p.metrics_predict_tf_with_crf(multiprocess_num=50,crf_config=crf_config,scales=[0.7,1.0,1.25])
        #p.metrics_predict_tf_with_crf(multiprocess_num=50,crf_config=crf_config)
        p.metrics_predict_tf_with_crf_max(multiprocess_num=50,crf_config=crf_config,scales=[0.5,0.75,1.0])
        #p.metrics_predict_tf_with_crf_max(multiprocess_num=50,crf_config=crf_config)
        end_time = time.time()
        print("total time:%f" % (end_time - start_time))

if __name__ == "__main__":
    category_num = 21
    t = Test({"batch_size":1,"input_size":None,"epoches":1,"category_num":category_num,"model_path":"./20180622-3-4/final-0","accum_num":1})
    t.predict()
