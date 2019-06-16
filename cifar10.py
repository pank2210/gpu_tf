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

from six.moves import urllib
import tensorflow as tf
import keras.layers as KL

import cifar10_input

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('no_classes', NUM_CLASSES,
                            """Number of images to process in a batch.""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.0065   # Initial learning rate.
INITIAL_LEARNING_RATE = 0.1   # Initial learning rate.

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

def conv_layer( _scope, inpt, filter_shape, stride, stddev=5e-2, wd=0.04):
  with tf.variable_scope(_scope) as scope:
    print("*****",_scope,"**conv layer**",inpt.get_shape(),"**filter**",filter_shape,"**stride**",stride)
    out_channels = filter_shape[3]
    '''
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME", use_cudnn_on_gpu=True)
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    '''
    kernel = _variable_with_weight_decay('weights',
                                         shape=filter_shape,
                                         stddev=stddev,
                                         wd=wd)
    conv = tf.nn.conv2d(inpt, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.1))
    conv = tf.nn.bias_add(conv, biases)
    #conv = BatchNorm(name=scope.name)(conv, training=False)
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.relu( conv, name=_scope)
     
    _activation_summary(conv)
     
    return conv

def residual_block( _scope, inpt, output_depth, down_sample=False, kernel=3, stride=2, projection=True, stddev=1e-2, wd=None):
  with tf.variable_scope(_scope) as scope:
    input_depth = inpt.get_shape().as_list()[3]
    filter_1, filter_2, filter_3 = output_depth
    if down_sample:
        filter_ = [1,kernel,kernel,1]
        stride_ = [1,stride,stride,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=stride_, padding='SAME')

    print("***",_scope,"**resnet block**",inpt.get_shape(),"**channels**",output_depth,"***stride*",stride)
    #conv = conv_layer( "%s-%s" % (_scope,'1'),inpt, [ 1, 1, input_depth, filter_1], stride=2, stddev, wd)
    conv = conv_layer( "%s-%s" % (_scope,'1'),inpt, [ 1, 1, input_depth, filter_1], stride=stride)
    conv = conv_layer( "%s-%s" % (_scope,'2'),conv, [ kernel, kernel, filter_1, filter_2], stride=1)
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
    with tf.variable_scope('conv0') as scope:
      conv1 = conv_layer( scope.name, layers[-1], [7, 7, inpt.get_shape()[-1], out_channels], stride=2)
      #conv1 = BatchNorm(name=scope.name)(conv1, training=False)
      #conv1 = tf.layers.batch_normalization(conv1)
      #conv1 = tf.nn.relu( conv1, name=scope.name)
      layers.append(conv1)

    with tf.variable_scope('pool0') as scope:
      filter_ = [1,5,5,1]
      stride_ = [1,2,2,1]
      pool0 = tf.nn.max_pool(layers[-1], ksize=filter_, strides=stride_, padding='SAME')
      #pool0 = tf.layers.BatchNormalization(pool0)
      layers.append(pool0)
      print("*****",scope.name,"**pool0**",pool0.get_shape())
    #'''
    
    in_channels = out_channels 
    out_channels = in_channels 
    for i in range (2):
      for j in range (num_conv):
        with tf.variable_scope('res%d_%d' % (i+1,j+1)) as scope:
            out_channels = in_channels * channels_increase_factor
            conv2_x = residual_block( scope.name + 'a', layers[-1], [in_channels,in_channels,out_channels], kernel=3, stride=2, projection=True)
            #in_channels = out_channels 
            #out_channels = in_channels * 4
            conv2 = residual_block( scope.name + 'b', conv2_x, [in_channels,in_channels,out_channels], kernel=3, stride=1, projection=False)
            #conv2 = residual_block(scope.name + '-2',conv2_x, out_channels, False)
            layers.append(conv2_x)
            layers.append(conv2)
            in_channels = out_channels 

        #assert conv2.get_shape().as_list()[1:] == [32, 32, 16]
     
   
    #assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    with tf.variable_scope('flatten') as scope:
        out = tf.layers.flatten( layers[-1], name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    #assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    fc_layers = 2
    fc_units_increment_factor = 2
    out_fc_units = 1024
    for i in range(fc_layers):
      in_fc_units = out_fc_units
      out_fc_units *=  1
      with tf.variable_scope('fc_' + str(i+1)) as scope:
        #out = KL.Dense( units=out_fc_units, activation='relu', name=scope.name)( inputs=layers[-1])
        out = tf.layers.dense( inputs=layers[-1], units=out_fc_units, activation='relu', name=scope.name)
        out = tf.layers.dropout( inputs=out, rate=0.2, seed=101, training=is_training)
        #out = KL.Dense( inputs=layers[-1], units=out_fc_units, activation='relu', name=scope.name)
        #out = BatchNorm(name=scope.name)(out, training=False)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    #assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    with tf.variable_scope('final') as scope:
        #out = KL.Dense( out, NUM_CLASSES, activation='softmax', name=scope.name)
        #out = KL.Dense( NUM_CLASSES, activation='softmax', name=scope.name)(layers[-1])
        out = tf.layers.dense( layers[-1], NUM_CLASSES, activation='softmax', name=scope.name)
        layers.append(out)
        print("*****",scope.name,out.get_shape())
     
    return layers[-1]


def inference1(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  print("images","**************",images.get_shape())
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 1, 16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)
    '''
    conv1 = tf.layers.conv2d( images, 64, 5, activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d( conv1, pool_size = 3, strides=2, padding='same',name='pool1')
  '''
  print("conv1","**************",conv1.get_shape())
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  print("pool1","**************",pool1.get_shape())
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
   
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 16, 16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)
  print("conv2","**************",conv2.get_shape())
   
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  '''
    conv2 = tf.layers.conv2d( pool1, 64, 3, activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d( conv2, pool_size = 3, strides=2, padding='same',name='pool2')
  '''
  print("pool2","**************",pool2.get_shape())
   
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    '''
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    reshape = tf.reshape(pool2, [FLAGS.batch_size,-1])
    print("local3.reshape","**************",reshape.get_shape())
    '''
    local3 = tf.contrib.layers.flatten(pool2)
    print("local3 flatened","**************",local3.get_shape())
    dim = local3.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 256],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local3)
    '''
    dim = local3.get_shape()[1].value
    local3 = tf.layers.dense(inputs=local3,units=512,activation=tf.nn.relu,
                               use_bias=True, 
                               bias_initializer=_variable_on_cpu('biases',[512],
                                   tf.constant_initializer(0.1)),
                               kernel_initializer = _variable_with_weight_decay('weights', 
                                          shape=[dim, 512],
                                          stddev=0.04, wd=0.004)
                              )
    '''
    print("local3","**************",local3.get_shape())

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[256, 256],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    print("local4","**************",local4.get_shape())
    _activation_summary(local4)
    '''
    local4 = tf.layers.dense(inputs=local3,units=512,activation=tf.nn.relu,
                               use_bias=True, 
                               bias_initializer=_variable_on_cpu('biases',[512],
                                   tf.constant_initializer(0.1)),
                               kernel_initializer = _variable_with_weight_decay('weights', 
                                          shape=[512, 512],
                                          stddev=0.04, wd=0.004)
                              )
    '''
    print("local4","**************",local4.get_shape())

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [256, NUM_CLASSES],
                                          stddev=1/256.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    print("softmax","**************",softmax_linear.get_shape())
    _activation_summary(softmax_linear)
    '''
    softmax_linear = tf.layers.dense(local4,NUM_CLASSES)
    '''
    print("softmax_linear","**************",softmax_linear.get_shape())

  return softmax_linear


def loss(logits, labels, loss_type='losses'):
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
  labels = tf.cast(labels, tf.int64)
  print("labels","================",labels.get_shape())
  print("logits","================",logits.get_shape())
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  print("cross_entropy","================",cross_entropy.get_shape())
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  print("cross_entropy_mean","================",cross_entropy_mean.get_shape())
   
  tf.add_to_collection( loss_type, cross_entropy_mean)
   
  #test_probs = tf.reduce_max(logits,1)
  #_, test_accu = tf.metrics.accuracy(labels=tf.argmax(labels,1),predictions=tf.argmax(logits,1))
  if loss_type != 'losses':
    _, test_accu = tf.metrics.accuracy(labels=tf.argmax(labels,1),
                                       predictions=tf.argmax(logits,1))
    preds = tf.argmax( logits, 1)
    probs = tf.reduce_max( logits, 1)
    tf.add_to_collection( 'test_accuracy', test_accu)
    tf.add_to_collection( 'test_preds', preds)
    tf.add_to_collection( 'test_probs', probs)
     
    return preds, probs
  else:
     
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
