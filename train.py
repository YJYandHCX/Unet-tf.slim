#!/usr/bin/env python
# coding: utf-8

# In[1]:


import DATA
import Net


# In[2]:


import tensorflow as tf


# In[3]:


import numpy as np


# In[4]:


slim = tf.contrib.slim


# In[5]:


data = DATA.CADF()
net = Net.Unet(True)


# In[6]:


global_step = tf.train.create_global_step()
learning_rate = 0.01
sess = tf.Session()
variable_to_restore = tf.global_variables()
saver = tf.train.Saver(variable_to_restore)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./Summ/",sess.graph,flush_secs=60)
tf.summary.scalar('learning_rate',learning_rate)
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(net.total_loss, global_step=global_step)
with tf.control_dependencies([train_step]):
    train_op = tf.no_op(name='train')
sess.run(tf.global_variables_initializer())


# In[ ]:


for i in range (1001):
    xs,ys = data.batch_prepare(i)
    feed_dict = {net.images:xs,net.labels:ys} 
    if i%10==0:
        summary_str,loss_value,_,step = sess.run([summary_op,net.total_loss,
                                                        train_op,global_step],feed_dict = feed_dict)
        writer.add_summary(summary_str,step)
        print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
    else :
        summary_str,loss_value,_,step = sess.run([summary_op,net.total_loss,train_op,global_step],
                                                 feed_dict = feed_dict)
    if i % 100 == 0:
        print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
        saver.save(sess,"./save/model.ckpt",global_step)
print("training is ending")


# In[9]:


sess.close()


# In[ ]:




