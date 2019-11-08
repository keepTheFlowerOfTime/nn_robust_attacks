from setup_cifar import load_batch,CifarFix
import numpy as np
import matplotlib.pyplot as plt
import math

from index_helper import color_grad,color_grad_reverse,draft,color_emblem,l2,random_point
from viewer import image_view
import skimage.io
import networkx
import matplotlib.patches as graphic


class Sensor:
    def __init__(self,fPaths:list,max_batch=10000):
        split=10

        all_data=[]
        all_labels=[]
        for f in fPaths:
            data,labels=load_batch(f,pre_fix='o')
            all_data.append(data)
            all_labels.append(labels)

        self.dataset=[np.concatenate(all_data),np.concatenate(all_labels)]

    def std(self,batch=1000):
        result=[self._sensor_std(x,y) for x,y in self.batches(batch)]

        return np.concatenate(result)

    def grad_std(self,batch=1000,level=1):
        result=[self._sensor_grad_std(x,y,level) for x,y in self.batches(batch)]
        
        return np.concatenate(result)

    

    def grad_statis(self,r='s'):
        result=[self._color_grad_statis(color_grad(x)) for x,_ in self.batches(10000)]
        if r=='c':
            return np.concatenate(result)
        else:
            t=np.zeros_like(result[0])
            for x in result:
                t+=x
            return t

    def main(self,batch=1000):
        return self._sensor_main(self.std(batch))

    def _sensor_main(self,index,ratio=0.7):
        #index:(n,class_number)向量
        class_number=10
        n=index.shape[0]
        split_class_index=np.split(index,class_number,1)
        result=[]
        for i in range(class_number):
            t=split_class_index[i]
            avg=np.average(t)

            t=np.abs(t-avg)
            t=np.sort(t)
            
            dev=t[min(int(n*ratio),n-1)]
            result.append([avg-dev,avg+dev])
        return np.array(result).reshape([-1,10])

    def _sensor_std(self,data,labels):
        std=[np.std(x) for x in data]
        std_result=[0]*labels.shape[1]
        for i,e in enumerate(std):
            std_result[np.argmax(labels[i])]+=e
        return np.array(std_result).reshape([1,-1])

    def _sensor_grad_std(self,data,labels,level=1):
        for i in range(level):
            data=color_grad(data)

        result=np.zeros_like(data[0])
        for i in range(data.shape[0]):
            result+=e

        return self._sensor_std(data,labels)

    def _color_grad_statis(self,color_grad):
        split=25
        input_arg=color_grad
        abs_1=np.abs(input_arg)
        divide_2=abs_1//split

        vec_3=np.reshape(divide_2,[-1])
        
        result=[0]*66
        for e in vec_3:
            result[int(e)]+=1

        return np.array(result).reshape([1,-1])

    def batches(self,number=1000):
        if(number<=0):
            return self.dataset[0],self.dataset[1]
        total=self.dataset[0].shape[0]
        split_number=total//number

        data_batches=np.split(self.dataset[0],split_number)
        labels_batches=np.split(self.dataset[1],split_number)

        for i in range(split_number):
            yield data_batches[i],labels_batches[i]

    def compute(self,index):
        data,labels=self.batches(0)
        total=data.shape[0]
        return np.array([index(x) for x in data])


    @staticmethod
    def eq(a:float,b:float)->bool:
        return math.isclose(a,b)

def draw_picture(f,i,d_sortby_type,core):
    ax=f.add_subplot(111)
    ratio=0.1
    most_index=int(d_sortby_type.shape[1]*0.7-1)
    min_distance=d_sortby_type[i][0]*ratio
    max_distance=d_sortby_type[i][d_sortby_type.shape[1]-1]*ratio
    most_distance=d_sortby_type[i][most_index]*ratio

    core_distance=[]
    for index in range(core.shape[0]):
        core_distance.append(np.linalg.norm(core[i]-core[index]))

    #draw min and max distance
    ax.add_patch(graphic.Circle(xy=(0,0),radius=max_distance,facecolor='green')) 
    ax.add_patch(graphic.Circle(xy=(0,0),radius=most_distance,facecolor='yellow'))
    ax.add_patch(graphic.Circle(xy=(0,0),radius=min_distance,facecolor='blue'))
    
    for i in range(core.shape[0]):
        ax.add_patch(graphic.Circle(xy=random_point((0,0),core_distance[i]*ratio),radius=2.5,facecolor='black'))
    #plt.axis('scaled')
    plt.axis('equal')
    plt.show()
data=CifarFix('o','o',need_verify_data=False,need_train_data=True)
train_data=np.concatenate([data.validation_data,data.train_data],axis=0)
labels=np.concatenate([data.validation_labels,data.train_labels],axis=0)
grad,emblem=color_emblem(train_data)

core=np.zeros([10,emblem.shape[1]])
core_count=np.zeros([10])
for i in range(emblem.shape[0]):
    type_index=np.argmax(labels[i])
    core[type_index]+=emblem[i]
    core_count[type_index]+=1

core/=(core_count.reshape([10,1]))

l2distance=l2(emblem,labels,core)

d_sortby_type=[[] for i in range(10)]
for i in range(train_data.shape[0]):
    d_sortby_type[np.argmax(labels[i])].append(l2distance[i])

d_sortby_type=np.array(d_sortby_type)
d_sortby_type=np.sort(d_sortby_type)

f=plt.figure()



def on_key_press(event):
    k=event.key
    try:
        k=int(k)
    except:
        k=-1
    if k==-1: return
    plt.cla()
    draw_picture(f,k,d_sortby_type,core)

f.canvas.mpl_connect('key_press_event', on_key_press)

draw_picture(f,0,d_sortby_type,core)
    





