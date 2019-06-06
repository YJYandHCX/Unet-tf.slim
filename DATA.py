#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import pickle


# In[20]:


class CADF():
    def __init__(self):
        self.image_size = 300
        self.pic_path = './train/json/'
        self.label_path = './train/label/'
        self.batch_size = 2
        self.num_channel = 3
        
    def im_read(self,read_path):
        im = cv2.imread(read_path)
        im = cv2.resize(im,(self.image_size,self.image_size))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = (im/255.0)*2.0-1.0
        return im
    
    def label_read(self,read_path):
        la_f = open(read_path,'rb')
        label = pickle.load(la_f)
        la_f.close()
        return label
        
    def batch_picture(self,start,end):
        images = np.zeros([self.batch_size,self.image_size,self.image_size,self.num_channel],dtype = np.float32)
        for i in range(self.batch_size):
            read_path = self.pic_path + str(i+start) + '.jpg'
            im = self.im_read(read_path)
            images[i,:,:,:] = im
        return images
    
    def batch_label(self,start,end):
        labels = np.zeros([self.batch_size,self.image_size,self.image_size],dtype = np.int32)
        for i in range(self.batch_size):
            read_path = self.label_path + str(i+start) + '.pkl'
            la = self.label_read(read_path)
            labels[i,:,:] = la
        return labels
    def batch_prepare(self,steps):
        start = steps*self.batch_size%200+1
        end = start + self.batch_size -1 
        #print (start)
        #print (end)
        images = self.batch_picture(start,end)
        labels = self.batch_label(start,end)
        return images,labels


# In[21]:


#ata = CADF()


# In[25]:


#import matplotlib.pyplot as plt
#images,labels = data.batch_prepare(40)


# In[28]:


#for i in range(5):
#    plt.imshow(images[i])
#    plt.show()


# In[29]:


#for i in range(5):
#    im = np.zeros([300,300,3])
#    for k in range(300):
#        for o in range(300):
#            if labels[i,k,o] == 1:
#                im[k,o,0] = 255
#    plt.imshow(im)
#    plt.show()


# In[ ]:




