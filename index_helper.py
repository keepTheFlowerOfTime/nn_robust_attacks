import numpy as np
import random
import math

def entropy(data,min_value=0,max_value=255,split_number=10,max_batch=3000,need_statistic=False):
    def _entropy(data,min_value,max_value):
        batch=data.shape[0]
        boundary=(max_value-min_value+1)//split_number
        if (max_value-min_value+1)%split_number!=0:
            boundary+=1

        statistic=[[0]*split_number for i in range(batch)]
        vec_data=np.reshape(data,[batch,-1])
        for i in range(batch):
            for e in vec_data[i]:
                statistic[i][int(e/boundary)]+=1
        statistic=np.array(statistic)
        prob=statistic/vec_data.shape[1]
        prob=np.where(prob!=.0,prob,1)
        result=[np.sum(-np.log(prob)*prob,axis=1,keepdims=True)]
        if need_statistic:
            result.append(statistic)
        
        return result
    """
    only support int type np.array,shpae=[b,-1]
    """
    batch=data.shape[0]
    if(batch>max_batch):
        batch_number=int(batch/max_batch)
        batches=np.split(data,batch_number)

        rt=[_entropy(x,min_value,max_value) for x in batches]
        return_number=len(rt[0])
        r=[]
        for i in range(return_number):
            r.append([x[i] for x in rt])
        
        for i in range(len(r)):
            r[i]=np.concatenate(r[i],axis=0)
    else:
        r=_entropy(data,min_value,max_value)
    return r

def for_each_class(data,labels,func,class_number=10):
    """
    func(class_index,data,previous_return)
    """
    b=data.shape[0]
    class_temp=[None]*class_number
    for i in range(b):
        class_index=np.argmax(labels[i])
        r=func(class_index,data[i],class_temp[class_index])
        class_temp[class_index]=r
    
    return class_temp

def for_each_batch(data,max_batch=3000):
    """
    see the first dimension of data as batch dimension.
    This function will split a big batch to some smaller batch.
    """
    b=data.shape[0]
    if b<=max_batch:
        yield data
    else:
        number=int(b/max_batch)
        if b%max_batch!=0: number+=1
        batches=np.split(data,number)
        for e in batches:
            yield e

def color_grad(data,ignore_boundary=False,max_batch=3000):
        kernal=np.array([
                [0,-1,0],
                [-1,2,0],
                [0,0,0]
                ])
        if len(data.shape)==3:
            data=data.reshape([1,data.shape[0],data.shape[1],data.shape[2]])
        b,w,h,channel=data.shape
        weight=get_weight(kernal,channel,channel)
        if b>max_batch:
            number=int(b/max_batch)
            batches=np.split(data,number)
            r_t=[conv2d_im2col(weight,x) for x in batches]
            r=np.concatenate(r_t,axis=0)
        else:
            r= conv2d_im2col(weight,data)
        if(ignore_boundary):
            r[:,0,:,:]=0
            r[:,:,0,:]=0
        return r

def color_emblem(data):
    """
    similar like(color_grad(data,True)),but also return the color which is erase
    """
    r=color_grad(data,False)
    b,w,h,c=r.shape

    row=np.zeros([b,h])
    col=np.zeros([b,w])

    for i in range(c):
        row+=(r[:,0,:,i].reshape([b,h]))
        col+=(r[:,:,0,i].reshape([b,w]))

    r[:,0,:,:]=0
    r[:,:,0,:]=0

    return r,np.concatenate([row,col],axis=1)

def color_compare(data,ignore_boundary=False,max_batch=3000):
    """
    data.shape=(b,w,h,c)
    """
    b,w,h,c=data.shape
    t=-1/9
    kernal=np.array([
            [t,t,t],
            [t,1+t,t],
            [t,t,t]
        ])
    weight=get_weight(kernal)
    if b>max_batch:
        number=int(b/max_batch)+1
        batches=np.split(data,number,axis=0)
        r=np.concatenate([conv2d_im2col(weight,x) for x in batches],axis=0)
    else:
        r=conv2d_im2col(weight,data)
    
    #r=data-r/8
    if ignore_boundary:
        r[:,0,:,:]=0
        r[:,:,0,:]=0
        r[:,w-1,:,:]=0
        r[:,:,h-1,:]=0

    return r

def l2(data,labels,core):
    """
    data.shape(b,w+h)
    labels.shape(b,typeNumber)
    core.shape(typeNumber,w+h)
    """
    b,emblem_length=data.shape
    core_ext=[core[np.argmax(labels[i])] for i in range(b)]
    core_ext=np.array(core_ext)

    return np.linalg.norm((data-core_ext),ord=2,axis=1)
        
def random_point(origin,distance):
    delta=random.random()*math.pi

    x=distance*math.cos(delta)
    y=distance*math.sin(delta)

    return x,y

def color_grad_reverse(color_grad):
        def forge_row(previous,grad):
            """
            shape:(n,1,height,channel)
            """
            result=np.zeros_like(grad)
            height=color_grad.shape[2]
            for i in range(height):
                if i==0:
                    result[:,:,i,:]=(previous[:,:,i,:]+grad[:,:,i,:])/2
                else:
                    result[:,:,i,:]=(result[:,:,i-1,:]+previous[:,:,i,:]+grad[:,:,i,:])/2
            
            return result
        #shape:(n,width,height,channel)
        width=color_grad.shape[1]
        grad_index=np.split(color_grad,width,1)
        previous=np.zeros_like(grad_index[0])
        
        result=[]
        for i in range(width):
            previous=forge_row(previous,grad_index[i])
            result.append(previous)

        return np.concatenate(result,2)

def conv2d_im2col(Weight,X,stride=1,padding='same'):
    def im2col(img, ksize, stride=1):
        N, H, W, C = img.shape
        out_h = (H - ksize) // stride + 1
        out_w = (W - ksize) // stride + 1
        col = np.empty((N * out_h * out_w, ksize * ksize * C))
        outsize = out_w * out_h
        for y in range(out_h):
            y_min = y * stride
            y_max = y_min + ksize
            y_start = y * out_w
            for x in range(out_w):
                x_min = x * stride
                x_max = x_min + ksize
                col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)
        return col
    Weight=np.transpose(Weight,[3,0,1,2])
    FN, ksize, ksize, C = Weight.shape
    if padding == 'same':
	    p = ksize // 2
	    X = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
    N, H, W, C = X.shape
    col = im2col(X, ksize, stride)
    z = np.dot(col, Weight.reshape(Weight.shape[0], -1).transpose())
    z = z.reshape(N, int(z.shape[0] / N), -1)
    out_h = (H - ksize) // stride + 1
    return z.reshape(N, out_h, -1 , FN)

def get_weight(kernal,in_channel=3,out_channel=3):
    def forge_row(zero,value,length,index):
        r=[zero]*length
        r[index]=value
        return r
    zero=np.zeros_like(kernal)
    t=[forge_row(zero,kernal,in_channel,i) for i in range(out_channel)]
    w=np.array(t)
    return np.transpose(w,[2,3,1,0])

def threhold(data,boundary):
    return np.where(data>boundary,data,0)

def focus(data,ignore=8,n=2):
    p=threhold(data,ignore)
    return np.power(p,2)

def area_attention(data):
    """
    return area,specify_value
    """
    color_use=np.sum(data)

    data=np.where(data<1,data,1)
    data=np.where(data==1,data,0)

    area=np.sum(data)

    return area,color_use/area

def draft(data,ratio,mode='b'):
    """
    mode: b return 0/1 image (default)
    mode: n return v/255-0.5 image
    """
    def get_lower_boundary(grad,need_number):
        batch,c,w,h=grad.shape
        vec_grad=np.reshape(grad,[batch,c,-1])
        length=vec_grad.shape[2]

        vec_abs_grad_view=np.abs(vec_grad)
        vec_abs_sort_grad_view=np.sort(vec_abs_grad_view,axis=2)
        lower_boundary=vec_abs_sort_grad_view[:,:,length-1-need_number:length-need_number]
        if mode=='n':
            r=np.where(vec_grad>lower_boundary,vec_grad/255-0.5,0)
        else:
            r=np.where(vec_grad>=lower_boundary,1,0)
        return r.reshape(batch,c,w,h)
    if(len(data)==4):
        b,w,h,c=data.shape
    else:
        w,h,c=data.shape
        b=1
        data=np.reshape(data,[b,w,h,c])
    area=w*h

    use_point_numner=int(area*ratio)

    grad=color_compare(data,True).transpose([0,3,1,2])
    r=get_lower_boundary(grad,use_point_numner)

    return r.transpose([0,2,3,1])
def split_area(data,ignore_area):
    pass

def try_focus(data):
    """
    data is a origin image,where shape is (width,height,channel)
    """
    width,height,channel=data.shape
    area=width*height
    ignore_area=16

    #first step: use color_grad to split the image into some region

def shuffle(data,rule=None,rng=None):
    """
    shuffle a n-dimension array,return result and rule
    """
    if rng==None:
        rng=np.random.default_rng()
    vData=data.reshape([-1])
    if rule is None:
        rule=np.arange(vData.shape[0])
        rng.shuffle(rule)

    t_vData=np.zeros_like(vData)
    t_vData[:]=vData[rule]
    return t_vData.reshape(data.shape),rule
def test():
    t=[
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ]

    t=np.array(t).reshape([1,3,3,1])
    #print(t[:,0,:,:])
    p=color_grad(t)
    print(p.reshape([-1]))
    p_r=color_grad_reverse(p)
    print(p_r.reshape([-1]))

def test_shuffle():
    data=np.arange(16).reshape([-1,4])
    print(data)
    rData,rule=shuffle(data)
    print(rData)
    print(rule)

#test_shuffle()