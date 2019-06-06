
# coding: utf-8

# In[22]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
#from labelme import utils
import json


# In[23]:


def json_to_png(name):
    name = str(name)
    path = './' + name + '.json'
    data = json.load(open(path))
    
    w = data['imageWidth']
    h = data['imageHeight']
    im = np.zeros([h,w,3],dtype=np.int32)
    
    
    shapes = data['shapes']
    oo = shapes[0]
    points = oo['points']
    num_p = len(points)
    
    p_points = np.zeros([1,num_p,2],dtype=np.int32)
    for i in range(num_p):
        p_points[0,i,:] = points[i]
    cv2.fillPoly(im, p_points, [0,0,255])
    
    #plt.imshow(im)
    #plt.show()
    
    s_name = name + '.png'
    cv2.imwrite('./png/'+s_name,im)
    return 0


# In[24]:


for i in range(1,11):
    _ = json_to_png(i)
print ("done")


# In[ ]:




