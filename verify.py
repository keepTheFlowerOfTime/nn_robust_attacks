## verify.py -- check the accuracy of a neural network
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from .setup_cifar import CIFAR, CIFARModel,CifarFix
from .setup_mnist import MNIST, MNISTModel
#from .setup_inception import ImageNet, InceptionModel

import tensorflow as tf
import numpy as np

import os

BATCH_SIZE = 1
Seed=[123456789,987654321,15247321,968373291,864928234,277182372,2837214283,283172394,8273824791,92374891,4283729121,
182372312,918282123,19237271823,283712831,5387319283,8237483912,6478347821,563489281,8391282391]

def test(sess,example):
    data,model=example
    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y=model.predict(x)
    r = []
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
    
    #print(np.mean(r))
    return np.array(r)

def predict(sess,data,model):
    r=[]
    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y=model.predict(x)
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        #print(pred)
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(pred)
    
    #print(np.mean(r))
    return np.concatenate(r)

def testCifar(sess):
    data,model=CifarFix('n','n',need_train_data=False),CIFARModel("models/cifar", sess)
    t=test(sess,(data,model))
    del data
    del model
    return t

def complete_result(override=True):
    seed=Seed
    for s in seed:
        model_path="models/cifar_shuffle_{}".format(s)
        result_path='{}.eva.npy'.format(model_path)

        if not os.path.exists(model_path): 
            print('{} not Exist!'.format_map(model_path))
            continue

        if not os.path.exists(result_path) or override:
            result=None
            with tf.Session() as sess:
                data,model=CifarFix('n','sn',need_train_data=False,seed=s),CIFARModel(model_path, sess)
                result=predict(sess,data,model)
            
            np.save(result_path,result)

def test_result():
    def p2classify(result,test_label):
        r=[]
        for i in range(len(test_label)):
            r.append(np.argmax(result[i]) == np.argmax(test_label[i]))
        return np.array(r)
    seed=Seed
    result=None
    eval_result_statistic=[]
    for s in seed:
        model_path="models/cifar_shuffle_{}".format(s)
        data=CifarFix('n','sn',need_train_data=False,seed=s)
        result_path='{}.eva.npy'.format(model_path)

        if not os.path.exists(model_path): 
            print('{} not Exist!'.format_map(model_path))
            continue
        
        if not os.path.exists(result_path): continue
        t=np.load(result_path,allow_pickle=True)
        if result is None: result=np.zeros_like(t)
        result+=t
        eval_result_statistic.append(p2classify(t,data.test_labels))

    eval_result=[]
    eval_result_class=[0]*10
    
    for i in range(0,len(data.test_data)):
        a=result[i]
        b=data.test_labels[i]

        p=np.argmax(a)==np.argmax(b)
        eval_result_class[np.argmax(b)]+=p

        eval_result.append(p)

    return np.array(eval_result),np.array(eval_result_class),eval_result_statistic
if __name__ == "__main__":
    complete_result(False)
    r1,r2,r3=test_result()
    print(np.mean(r1))
    print(r2/1000)

    t=np.zeros_like(r3[0])
    for e in r3:
        t&=e
    print(np.mean(t))
    # seed=Seed
    # result=[0]*len(seed)
    # for i in range(len(seed)):
        
    #     path="models/cifar_shuffle_{}".format(seed[i])
    #     if os.path.exists('{}.eva.npy'.format(path)):
    #         result[i]=np.load('{}.eva.npy'.format(path),allow_pickle=True)
    #     else:
    #         a=CifarFix('sn','sn',False,True,seed[i])
    #         with tf.Session() as sess:
    #             b=CIFARModel(path,sess)
    #             result[i]=test(sess,(a,b))

    #         del a

    # with tf.Session() as sess:
    #     result.append(testCifar(sess))

    # for i in range(len(seed)):
    #     print('seed:{}--prob:{}'.format(seed[i],np.mean(result[i])))

    # best_pro_array=[]
    # best_pro=np.zeros_like(result[0])
    # for e in result:
    #     best_pro|=e
    #     best_pro_array.append(np.mean(best_pro))
    # print('best prob:{}'.format(best_pro_array))
    
    # stack_prob=0
    # for i in range(0,len(result)-1):
    #     a=result[i]
    #     b=result[i+1]
    #     c=np.abs(a^b)
    #     stack_prob+=(np.mean(a)+np.mean(b)-np.mean(c))/2
    
    # print('stack prob:{}'.format(stack_prob/(len(result)-1)))


    # most_pro=np.zeros_like(result[0],dtype=np.int)
    # for e in result:
    #     most_pro+=e
    
    # t=np.where(most_pro>=5,most_pro,0)
    # t=np.where(a<5,a,1)
    # print('most_pro: {}'.format(np.mean(t)))
