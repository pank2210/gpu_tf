# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow.keras.layers as KL

import cifar10_input

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('no_classes', NUM_CLASSES,
                            """Number of images to process in a batch.""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.0065   # Initial learning rate.
INITIAL_LEARNING_RATE = 0.01   # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  #print("@@@",name,"@@@",shape,"@@@",stddev,"@@@",wd,"@@@")
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def maxpool_layer( inpt, filter_, stride_, padding_='SAME'):
    return tf.nn.max_pool(inpt, ksize=filter_, strides=stride_, padding=padding_)

def gavgpool_layer( inpt, filter_, stride_, padding_='SAME'):
    return tf.layers.average_pooling2d(inpt, pool_size=filter_, strides=stride_, padding=padding_)

def avgpool_layer( inpt, filter_, stride_, padding_='SAME'):
    return tf.nn.avg_pool(inpt, ksize=filter_, strides=stride_, padding=padding_)

def conv_layer( _scope, inpt, filter_shape, stride, stddev=5e-2, wd=0.04):
  with tf.variable_scope(_scope) as scope:
    print("*****",_scope,"**conv layer**",inpt.get_shape(),"**filter**",filter_shape,"**stride**",stride)
    out_channels = filter_shape[3]
    n = filter_shape[0] * filter_shape[1] * out_channels
    
    kernel = _variable_with_weight_decay('weights',
                                         shape=filter_shape,
                                         stddev=np.sqrt(2.0/n),
                                         wd=wd)
    conv = tf.nn.conv2d(inpt, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.1))
    conv = tf.nn.bias_add(conv, biases)
    #conv = BatchNorm(name=scope.name)(conv, training=False)
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.relu( conv, name=_scope)
     
    _activation_summary(conv)
     
    return conv

def inception_block( _scope, inpt, output_depths, conv_filters=[3,5], conv_strides=[2,2], pool_filter=3, pool_stride=1):
  with tf.variable_scope(_scope) as scope:
    input_depth = inpt.get_shape().as_list()[3]
    out_channel_1x1, reduce_channel_3x3, out_channel_3x3, reduce_channel_5x5, out_channel_5x5, pool_proj = output_depths
     
    print("*****",_scope,"**inception_block**",inpt.get_shape(),"**cfilter**",conv_filters,"**cstride**",conv_strides,"**pfilter**",pool_filter,"**pstride**",pool_stride)
     
    #Conv1X1 layer of inception block 
    print("**",_scope,"***1st conv block***")
    conv1x1 = conv_layer( _scope + '_a1x1', inpt, filter_shape=[1,1,input_depth, out_channel_1x1], stride=1)
     
    #Conv3X3 layer of inception block 
    print("**",_scope,"***2nd conv block***")
    conv3x3 = conv_layer( _scope + '_b1x1', inpt, filter_shape=[1,1,input_depth, reduce_channel_3x3], stride=1)
    conv3x3 = conv_layer( _scope + '_b3x3', conv3x3, filter_shape=[conv_filters[0],conv_filters[0],reduce_channel_3x3, out_channel_3x3], stride=conv_strides[0])
     
    #Conv5X5 layer of inception block 
    print("**",_scope,"***3rd conv block***")
    conv5x5 = conv_layer( _scope + '_c1x1', inpt, filter_shape=[1,1,input_depth, reduce_channel_5x5], stride=1)
    conv5x5 = conv_layer( _scope + '_c5x5', conv5x5, filter_shape=[conv_filters[1],conv_filters[1],reduce_channel_5x5, out_channel_5x5], stride=conv_strides[1])
     
    #Pool layer of inception block 
    print("**",_scope,"***max_pool block***")
    #pool3x3 = maxpool_layer( inpt, filter_=[1,pool_filter,pool_filter,1], stride_=[1,pool_stride,pool_stride,1], padding_='SAME')
    pool3x3 = avgpool_layer( inpt, filter_=[1,pool_filter,pool_filter,1], stride_=[1,pool_stride,pool_stride,1], padding_='SAME')
    print("*****",_scope,"**maxpool layer**",pool3x3.get_shape(),"**filter**",pool_filter,"**stride**",pool_stride)
    pool3x3 = conv_layer( _scope + '_d1x1', pool3x3, filter_shape=[1,1,input_depth, pool_proj], stride=1)
     
    out = tf.concat([conv1x1,conv3x3,conv5x5,pool3x3],axis=3,name=_scope + '_concat') 
    print("******",_scope,"***stack***",out.get_shape().as_list(),"****")
    
    return out

def residual_block( _scope, inpt, output_depth, down_sample=False, kernel=3, stride=2, projection=True, stddev=1e-2, wd=None):
  with tf.variable_scope(_scope) as scope:
    input_depth = inpt.get_shape().as_list()[3]
    filter_1, filter_2, filter_3 = output_depth
    if down_sample:
        filter_ = [1,kernel,kernel,1]
        stride_ = [1,stride,stride,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=stride_, padding='SAME')

    print("***",_scope,"**resnet block**",inpt.get_shape(),"**channels**",output_depth,"***stride*",stride)
    conv = conv_layer( "%s-%s" % (_scope,'1'),inpt, [ 1, 1, input_depth, filter_1], stride=1)
    conv = conv_layer( "%s-%s" % (_scope,'2'),conv, [ kernel, kernel, filter_1, filter_2], stride=stride)
    conv = conv_layer( "%s-%s" % (_scope,'3'),conv, [ 1, 1, filter_2, filter_3], stride=1)
     
    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            _scope_name = "%s-%s" % (_scope,'R')
            input_layer = conv_layer( "%s-%s" % (_scope,'R'),inpt, [1, 1, input_depth, filter_3], stride=stride)
            #input_layer = BatchNorm(name=_scope_name)(input_layer,training=False)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, filter_3 - input_depth]])
    else:
        input_layer = inpt
     
    print("***",_scope,"**Add resnet block**",conv.get_shape(),"***",input_layer.get_shape())
    conv = conv + input_layer
    conv = tf.nn.relu( conv, name=_scope)
     
    return conv

# Inception Net architectures used for CIFAR-10
def inception(inpt, is_training=True):
    layers = []
    layers1 = []
   
    enable_pooling = True
    channels_increase_factor = 2
    layers.append(inpt)  #0
    print("**inpt**",inpt.get_shape(),"***")
    
    out_channels = 64
    with tf.variable_scope('1a') as scope:
      out = conv_layer( scope.name, layers[-1], [7, 7, inpt.get_shape()[-1], out_channels], stride=2)
      layers.append(out) #1
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      #out = avgpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out) #2
      print("*****",scope.name,"**out**",out.get_shape())
     
    out_channels = 192
    with tf.variable_scope('1b') as scope:
      out = conv_layer( scope.name, layers[-1], [3, 3, layers[-1].get_shape()[-1], out_channels], stride=1)
      layers.append(out) #3
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      #out = avgpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out) #4
      print("*****",scope.name,"**out**",out.get_shape())
     
    ''' 
    out_channels = 192
    with tf.variable_scope('2a') as scope:
      out = conv_layer( scope.name, layers[-1], [3, 3, layers[-1].get_shape()[-1], out_channels], stride=1)
      layers.append(out) #5
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      #out = avgpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out) #6
      print("*****",scope.name,"**out**",out.get_shape())
    ''' 
     
    ''' 
    #Layer 1a out channels=256 
    #out_channel_1x1, reduce_channel_3x3, out_channel_3x3, reduce_channel_5x5, out_channel_5x5, pool_proj = output_depths
    output_channels = [8,16,16,16,16,8] 
    with tf.variable_scope('1a') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=5, 
                             pool_stride=1)
      layers.append(out)
     
    #Max pool layer 
    with tf.variable_scope('maxpool1a') as scope:
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out)
    ''' 
     
    #Layer 3a out channels=256 
    #out_channel_1x1, reduce_channel_3x3, out_channel_3x3, reduce_channel_5x5, out_channel_5x5, pool_proj = output_depths
    output_channels = [64,96,128,16,32,32] 
    with tf.variable_scope('3a') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out) #7
     
    #Layer 3b out channels=480 
    output_channels = [128,128,192,32,96,64] 
    with tf.variable_scope('3b') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out) #8
     
    #Max pool layer 
    with tf.variable_scope('maxpool3b') as scope:
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out) #11
     
    #''' 
    #Layer 4a out channels=512 
    output_channels = [192,96,208,16,48,64] 
    with tf.variable_scope('4a') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out) #12
     
    #Layer 4b out channels=512 
    output_channels = [160,112,224,24,64,64] 
    with tf.variable_scope('4b') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
    #Layer 4c out channels=512 
    output_channels = [128,128,256,24,64,64] 
    with tf.variable_scope('4c') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
     
    #Layer 4d out channels=528 
    output_channels = [112,144,288,32,64,64] 
    with tf.variable_scope('4d') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
    #Layer 4e out channels=528 
    output_channels = [256,160,320,32,128,128] 
    with tf.variable_scope('4e') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
    #Max pool layer 
    with tf.variable_scope('maxpool4e') as scope:
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = maxpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out)
     
    #Layer 5a out channels=512 
    output_channels = [256,160,320,32,128,128] 
    with tf.variable_scope('5a') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
    #Layer 5b out channels=512 
    output_channels = [384,192,384,48,128,128] 
    with tf.variable_scope('5b') as scope:
      out = inception_block( scope.name, 
                             inpt=layers[-1], 
			     output_depths=output_channels,
                             conv_filters=[3,5], 
                             conv_strides=[1,1], 
                             pool_filter=3, 
                             pool_stride=1)
      layers.append(out)
     
    #AvgGlobal pool layer 
    with tf.variable_scope('avgglobalpool5b') as scope:
      #filter_ = [1,3,3,1]
      filter_ = [7,7]
      stride_ = [1,1]
      print("*****",scope.name,"**avgglobalool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      out = gavgpool_layer(layers[-1], filter_, stride_, padding_='SAME')
      layers.append(out)
     
    #''' 
     
    #Flatten layer
    with tf.variable_scope('flatten_and_dropout') as scope:
        out = tf.layers.flatten( layers[-1], name=scope.name)
        out = tf.layers.dropout( inputs=out, rate=0.4, seed=101, training=is_training)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
    
    ''' 
    #final FC layer
    fc_layers = 2
    fc_units_increment_factor = 2
    out_fc_units = 512
    for i in range(fc_layers):
      in_fc_units = out_fc_units
      out_fc_units *=  1
      with tf.variable_scope('fc_final_' + str(i)) as scope:
        out = tf.layers.dense( inputs=layers[-1], units=out_fc_units, activation='relu', use_bias=True, kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0), bias_initializer=tf.constant_initializer(), name=scope.name)
        out = tf.layers.dropout( inputs=out, rate=0.5, seed=101, training=is_training)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
    '''
     
    #Final sigmoid Layer
    with tf.variable_scope('final') as scope:
        out = tf.layers.dense( layers[-1], NUM_CLASSES, activation='sigmoid', name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    #return layers[-1], layers1[-1], layers2[-1]
    return layers[-1]



n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet(inpt, n, is_training=True):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    layers = []
   
    enable_pooling = True
    out_channels = 64
    channels_increase_factor = 2
    layers.append(inpt) 
    print("**inpt**",inpt.get_shape(),"***num_conv***",num_conv)
    #'''
    with tf.variable_scope('conv01') as scope:
      conv1 = conv_layer( scope.name, layers[-1], [5, 5, inpt.get_shape()[-1], out_channels], stride=1)
      layers.append(conv1)

    with tf.variable_scope('pool01') as scope:
      filter_ = [1,5,5,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      pool0 = tf.nn.max_pool(layers[-1], ksize=filter_, strides=stride_, padding='SAME')
      layers.append(pool0)
      print("*****",scope.name,"**pool0**",pool0.get_shape())
    
    in_channels = out_channels 
    out_channels = 128 
    with tf.variable_scope('conv02') as scope:
      conv1 = conv_layer( scope.name, layers[-1], [3, 3, in_channels, out_channels], stride=1)
      layers.append(conv1)

    with tf.variable_scope('pool02') as scope:
      filter_ = [1,3,3,1]
      stride_ = [1,2,2,1]
      print("*****",scope.name,"**maxpool layer**",layers[-1].get_shape(),"**kernel**",filter_,"**stride**",stride_)
      pool0 = tf.nn.max_pool(layers[-1], ksize=filter_, strides=stride_, padding='SAME')
      layers.append(pool0)
      print("*****",scope.name,"**pool0**",pool0.get_shape())
    #'''
    
    in_channels = out_channels 
    out_channels = in_channels 
    for i in range (2):
      for j in range (num_conv):
        with tf.variable_scope('res%d_%d' % (i+1,j+1)) as scope:
            out_channels = in_channels * channels_increase_factor
            conv2 = residual_block( scope.name + 'a', layers[-1], [in_channels,in_channels,out_channels], down_sample=False, kernel=3, stride=2, projection=True)
            layers.append(conv2)
            conv2 = residual_block( scope.name + 'b', conv2, [in_channels,in_channels,out_channels], down_sample=False, kernel=3, stride=1, projection=False)
            layers.append(conv2)
            conv2 = residual_block( scope.name + 'c', conv2, [in_channels,in_channels,out_channels], down_sample=False, kernel=3, stride=1, projection=False)
            layers.append(conv2)
            in_channels = out_channels 

    with tf.variable_scope('flatten') as scope:
        out = tf.layers.flatten( layers[-1], name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    #assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    fc_layers = 2
    fc_units_increment_factor = 2
    out_fc_units = 512
    for i in range(fc_layers):
      in_fc_units = out_fc_units
      out_fc_units *=  1
      with tf.variable_scope('fc_' + str(i+1)) as scope:
        #out = KL.Dense( units=out_fc_units, activation='relu', name=scope.name)( inputs=layers[-1])
        out = tf.layers.dense( inputs=layers[-1], units=out_fc_units, activation='relu', use_bias=True, kernel_initializer=tf.uniform_unit_scaling_initializer(factor=1.0), bias_initializer=tf.constant_initializer(), name=scope.name)
        out = tf.layers.dropout( inputs=out, rate=0.5, seed=101, training=is_training)
        #out = KL.Dense( inputs=layers[-1], units=out_fc_units, activation='relu', name=scope.name)
        #out = BatchNorm(name=scope.name)(out, training=False)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    ''' 
    #assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    with tf.variable_scope('final') as scope:
        #out = KL.Dense( out, NUM_CLASSES, activation='softmax', name=scope.name)
        #out = KL.Dense( NUM_CLASSES, activation='softmax', name=scope.name)(layers[-1])
        out = tf.layers.dense( layers[-1], NUM_CLASSES, activation='softmax', name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
    ''' 
     
    #Final sigmoid Layer
    with tf.variable_scope('final') as scope:
        out = tf.layers.dense( layers[-1], NUM_CLASSES, activation='sigmoid', name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    return layers[-1]

def loss(logits, labels, loss_type='losses', threshold=.5):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  print("labels ================",labels.get_shape())
  print("logits ================",logits.get_shape())
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  print("cross_entropy","================",cross_entropy.get_shape())
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  print("cross_entropy_mean","================",cross_entropy_mean.get_shape())
   
  if loss_type != 'losses':
    tf.add_to_collection( loss_type, cross_entropy_mean)
    '''
    _, test_accu = tf.metrics.accuracy(labels=tf.argmax(labels,1),
                                           predictions=tf.argmax(logits,1))
    preds = tf.argmax( logits, 1)
    probs = tf.reduce_max( logits, 1)
    preds = logits
    preds[preds >= threshold] = 1.
    preds[preds < threshold] = 0.
    #print("preds","================",preds.get_shape())
    test_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels,tf.float32), logits), tf.float32))
    '''
    test_accu = 1 - tf.reduce_mean(tf.math.abs(tf.math.subtract(logits,tf.cast(labels,tf.float32))))
    #print("test_accu","================",test_accu.get_shape())
    tf.add_to_collection( 'test_accuracy', test_accu)
    #tf.add_to_collection( 'test_preds', preds)
    tf.add_to_collection( 'test_probs', logits)
    tf.add_to_collection( 'test_labels', labels)
     
    return logits
  else:
    tf.add_to_collection( loss_type, cross_entropy_mean)
     
    return tf.add_n(tf.get_collection(loss_type), name='total_' + loss_type)

def _add_loss_summaries(total_loss, loss_type='losses'):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection(loss_type)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
