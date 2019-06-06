#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import DATA
import Net
import config as cfg


# In[2]:


slim = tf.contrib.slim


# In[3]:


data = DATA.CADF()
net = Net.Unet(True)
global_step = tf.train.create_global_step()
learning_rate = 0.0001


# In[4]:


tf.summary.scalar('learning_rate',learning_rate)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(net.total_loss, global_step=global_step)
sess = tf.Session()
variable_to_restore = tf.global_variables()
saver = tf.train.Saver(variable_to_restore)
ckpt = tf.train.get_checkpoint_state('./save/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("OK")
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./Summ/",sess.graph,flush_secs=60)

with tf.control_dependencies([train_step]):
    train_op = tf.no_op(name='train')


# In[ ]:


for i in range (205):
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


# In[ ]:




