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

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10

import utils.data_util as du

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 15000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def tower_loss(scope, images, labels, loss_type='losses'):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  #logits = cifar10.inference1(images)
  logits = cifar10.resnet(inpt=images,n=20)
  #logits = cifar10.simple_net(inpt=images)
  #print("logits ############",logits.get_shape())

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels, loss_type)
  #print("After loss....")

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection( loss_type, scope)
  #losses = tf.get_collection('losses')
  #print("losses len ############",len(losses))

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_' + loss_type)
  #print("total_loss ############",total_loss.get_shape())
  
  if loss_type != 'losses':
    accuracies = tf.get_collection( 'test_accuracy', scope)
    mean_accuracy = tf.reduce_mean( accuracies)
     
    return total_loss, mean_accuracy
  else:
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
      #print("loss_name - ",loss_name)
      tf.summary.scalar(loss_name, l)

    return total_loss, None


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  i = 0
  j = 0
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #print(i,"grad_and_vars ############ len ",len(grad_and_vars))
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      #print("grad_and_vars ############ g ",g.get_shape())
      expanded_g = tf.expand_dims(g, 0)
      #print(j,"grad_and_vars ############ expanded_g ",expanded_g.get_shape())

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
      j += 1

    # Average over the 'tower' dimension.
    #print("grads ############ grad len ",len(grads))
    grad = tf.concat(axis=0, values=grads)
    #print("grad ############  ",grad.get_shape())
    grad = tf.reduce_mean(grad, 0)
    #print("grad ############  ",grad.get_shape())
    i += 1

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size / FLAGS.num_gpus)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    '''
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    '''
    data = du.Data(jfilepath='config/config.json')
    data.set_batch_size( FLAGS.batch_size)
    data.set_no_classes( FLAGS.no_classes)
    
    #training dataset
    data.set_data_file('train')
    _dataset,_iterator = data.get_iterator()
    training_init_op = _iterator.make_initializer(_dataset)
    
    #training dataset
    data.set_data_file('test')
    _test_dataset, _test_iterator = data.get_iterator()
    test_init_op = _test_iterator.make_initializer(_test_dataset)
    
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = _iterator.get_next()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss, _ = tower_loss(scope, image_batch, label_batch)
            print("scope - ",scope,"loss #############",loss.get_shape())

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
             
            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)
             
            #validation/Test set 
            loss_type = 'test_losses'
            test_image_batch, test_label_batch = _test_iterator.get_next()
            test_loss, test_accu = tower_loss(scope, test_image_batch, test_label_batch, loss_type)
            '''
            print("grads ############# len ",len(grads))
            for i,grad in enumerate(grads):
              for j,g in enumerate(grad):
                print(i,j,"grad +++",g)
            print("grads #############",grads.get_shape())
            '''

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            #print("tower_grads ############# len ",len(tower_grads))

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('tower_loss', loss))
    summaries.append(tf.summary.scalar('test_accu', test_accu))
    summaries.append(tf.summary.scalar('test_loss', test_loss))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    local_var_init = tf.local_variables_initializer()
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    sess.run(local_var_init)
    sess.run(training_init_op)
    sess.run(test_init_op)
     
    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        test_accu_value, test_loss_value = sess.run([test_accu, test_loss])
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f '
                      'sec/batch) test loss[%.4f] test accu[%.2f]')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, 
                             test_loss_value, test_accu_value))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
