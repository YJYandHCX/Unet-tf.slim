import tensorflow as tf
import config as cfg
import numpy as np
slim = tf.contrib.slim
class Unet():
    def __init__(self,is_training=True):
        self.classes = cfg.CLASSES
        self.class_ind = cfg.CLASSES_ind
        self.num_channel = cfg.NUM_CHANNEL
        self.num_class = cfg.NUM_CLASS
        self.image_size = cfg.IMAGE_SIZE
        self.phase = is_training
        
        self.images = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.num_channel])
        self.pre,self.conv9 = self.u_net(self.images,output_num=self.num_class)
        self.labels = tf.placeholder(tf.int32,[None,self.image_size,self.image_size])        
        if self.phase:
            self.loss_layer(self.conv9,self.labels)
            self.total_loss = tf.losses.get_total_loss() 
            tf.summary.scalar('total_loss',self.total_loss)
    def u_net(self,image,output_num):
        with tf.variable_scope("u_net", reuse=None):
            with slim.arg_scope([slim.conv2d,  slim.conv2d_transpose, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # slim.conv2d default relu activation
                # subsampling
                conv0 = slim.repeat(image, 2, slim.conv2d, 32, [3, 3], scope='conv0')
                pool0 = slim.max_pool2d(conv0, [2, 2], scope='pool0')  # 1/2
                bn0 = slim.batch_norm(pool0, decay=0.9, epsilon=1e-5, scope="bn0")
            
                conv1 = slim.repeat(bn0, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')  # 1/4
                bn1 = slim.batch_norm(pool1, decay=0.9, epsilon=1e-5, scope="bn1")
            
                conv2 = slim.repeat(bn1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')  # 1/8
                bn2 = slim.batch_norm(pool2, decay=0.9, epsilon=1e-5, scope="bn2")
            
                conv3 = slim.repeat(bn2, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')  # 1/16
                bn3 = slim.batch_norm(pool3, decay=0.9, epsilon=1e-5, scope="bn3")
            
                conv4 = slim.repeat(bn3, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')  # 1/32
                bn4 = slim.batch_norm(pool4, decay=0.9, epsilon=1e-5, scope="bn4")
            
                # upsampling
                conv_t1 = slim.conv2d_transpose(bn4, 256, [2,2], scope='conv_t1') # up to 1/16 + conv3
                merge1 = tf.concat([conv_t1, conv3], 3)
                conv5 = slim.stack(merge1, slim.conv2d, [(512, [3, 3]),(256, [3,3])], scope='conv5')
                bn5 = slim.batch_norm(conv5, decay=0.9, epsilon=1e-5, scope='bn5')
            
                conv_t2 = slim.conv2d_transpose(bn5, 128, [2,2], scope='conv_t2') # up to 1/8 + conv2
                merge2 = tf.concat([conv_t2, conv2], 3)
                conv6 = slim.stack(merge2, slim.conv2d, [(256, [3,3]), (128, [3,3])], scope='conv6')
                bn6 = slim.batch_norm(conv6, decay=0.9, epsilon=1e-5, scope='bn6')
            
                conv_t3 = slim.conv2d_transpose(bn6, 64, [2,2], scope='conv_t3') # up to 1/4 + conv1
                merge3 = tf.concat([conv_t3, conv1], 3)
                conv7 = slim.stack(merge3, slim.conv2d, [(128, [3,3]), (64, [3,3])], scope='conv7')
                bn7 = slim.batch_norm(conv7, decay=0.9, epsilon=1e-5, scope='bn7')
            
                conv_t4 = slim.conv2d_transpose(bn7, 32, [2,2], scope='convt4')  # up to 1/2 + conv0
                merge4 = tf.concat([conv_t4, conv0], 3)
                conv8 = slim.stack(merge4, slim.conv2d, [(64, [3,3]), (32, [3,3])], scope='conv8')
                bn8 = slim.batch_norm(conv7, decay=0.9, epsilon=1e-5, scope='bn8')
            
                # output layer scoreMap
                conv9 = slim.conv2d(bn7,output_num, [1,1], scope='scoreMap') # 2 CLASSES_NUM
                annotation_pred = tf.argmax(conv9, dimension=3, name='prediction')
                return annotation_pred, conv9
    def loss_layer(self,logits,labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        class_loss = tf.reduce_mean(cross_entropy)
        regu_loss = tf.losses.get_regularization_loss()
        tf.losses.add_loss(regu_loss)
        tf.losses.add_loss(class_loss)
        tf.summary.scalar('regu_loss',regu_loss) 
        tf.summary.scalar('class_loss',class_loss)
