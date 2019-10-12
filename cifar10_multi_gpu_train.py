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
import os
import os.path
import re
import time

import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10

import utils.data_util as du

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/disk1/data1/data/models/inception',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1810000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/disk1/data1/data/models/inception',
                           """Directory where to read model checkpoints.""")

def tower_loss( scope, images, labels, loss_type='losses', image_ids=None):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  #set is_training flag for use in resnet for activating or deactivating dropout based on training
  is_training = True 
  if loss_type != 'losses':
    is_training = False
  #logits = cifar10.resnet(inpt=images,n=44,is_training=is_training)
  #logits = cifar10.inception(inpt=images,is_training=is_training)
  logits = cifar10.inceptionV3(inputs=images,is_training=is_training)
  print("logits ############",logits.get_shape())

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  if loss_type != 'losses':
    probs = cifar10.loss(logits, labels, loss_type)
  else:
    _ = cifar10.loss(logits, labels, loss_type)
  print("loss_type[%s]...." % (loss_type))

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection( loss_type, scope)
  #losses = tf.get_collection('losses')
  print("losses len ############",len(losses))
   
  if loss_type != 'losses':
    if image_ids is not None:
      tf.add_to_collection( 'image_id', image_ids)
      tf.add_to_collection('labels',labels)
      tf.add_to_collection('probs',probs)
    accuracies = tf.get_collection( 'test_accuracy', scope)
    mean_accuracy = tf.reduce_mean( accuracies)
     
    return mean_accuracy, probs
  else:
    
    #Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_' + loss_type)
    print("total_loss ############",total_loss.get_shape())
  
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
      print("loss_name - ",loss_name)
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
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #print(i,"grad_and_vars ############ len ",len(grad_and_vars))
    j = 0
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      #print(i,j,"grad_and_vars ############ g ",g.get_shape())
      expanded_g = tf.expand_dims(g, 0)
      #print(i,j,"grad_and_vars ############ expanded_g ",expanded_g.get_shape())
       
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


def train(model_name='mymodel.ckpt'):
  """Train CIFAR-10 for a number of steps."""
  print("Starting Train for model_name[%s]" % (model_name))
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf.random.set_random_seed(1001)
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
    #opt = tf.train.RMSPropOptimizer(lr)

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
    _dataset, _iterator = data.get_iterator2()
    training_init_op = _iterator.make_initializer(_dataset)
    
    #test dataset
    data.set_data_file('val')
    _test_dataset, _test_iterator = data.get_iterator2()
    test_init_op = _test_iterator.make_initializer(_test_dataset)
    
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            _, image_batch, label_batch = _iterator.get_next()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            print("Entering sope - ",scope)
            loss, _ = tower_loss(scope, image_batch, label_batch)
            print("scope - ",scope,"loss #############",loss.get_shape())

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
             
            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)
             
            #validation/Test set 
            _, test_image_batch, test_label_batch = _test_iterator.get_next()
            test_accu, test_probs = tower_loss(scope, test_image_batch, test_label_batch, loss_type='test_losses')
            
            #below code prints gradiants from the networks.
            '''
            print("grads ############# len ",len(grads))
            for i,grad in enumerate(grads):
              for j,g in enumerate(grad):
                print(i,j,"grad +++",g)
            '''
             
            # Keep track of the gradients across all towers.
            if grads is None:
              print("$$$$$$$$$$$$$$$$$$$$$$$$grads is None scope[%s]..." % (scope))
            tower_grads.append(grads)
            #print("tower_grads ############# len ",len(tower_grads))

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))
    summaries.append(tf.summary.scalar('tower_loss', loss))
    #for i,test_accu in enumerate(test_accus):
    #  summaries.append(tf.summary.scalar('test_accu' + str(i), test_accu))
    summaries.append(tf.summary.scalar('test_accu', test_accu))
    #summaries.append(tf.summary.scalar('test_loss', test_loss))

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
        test_label_value,test_accu_value, test_probs_value = sess.run([test_label_batch,test_accu,test_probs])
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus
        
        '''
        #print("*******test_accu[%s]*******" % (test_accu_value)) 
        #print("*******test_probs type*******", type(test_probs_value))
        #print("*******test_probs shape*******", test_probs_value.shape)
        print("*******test_probs value*******", test_probs_value[:5,:10])
         
        pt50 = test_probs_value 
        pt50[pt50 >= .5 ] = 1.
        pt50[pt50 < .5 ] = 0.
        pt50_accu = 1 - np.abs(pt50 - test_label_value).mean()
         
        print("*******test_label_value *******", test_label_value.shape)
        print("*******test_label_value value*******", test_label_value[:5,:10])
        #print("*******pt50 *******", pt50.shape)
        print("*******pt50 value*******", pt50[:5,:10])
        '''
         
        #calculate pred accuracy for parametric target 
        pt50_accu = np.abs(test_label_value - test_probs_value) < .001 
        pt50_accu = pt50_accu.sum()/pt50_accu.size
         
        format_str = ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f '
                      'sec/batch) accu[%.5f]')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, 
                             pt50_accu))
         
        ''' 
        if type(test_accu_value) is list: 
          format_str = ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f '
                      'sec/batch)  accu0[%.2f] accu1[%.2f] accu2[%.2f]')
          print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, 
                             test_accu_value[0],test_accu_value[1],test_accu_value[2]))
        else: 
          format_str = ('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f '
                      'sec/batch)  test accu[%.2f]')
          print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch, 
                             test_accu_value))
        ''' 

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        checkpoint_path = os.path.join(FLAGS.train_dir, model_name)
        saver.save(sess, checkpoint_path, global_step=step)

def test2(model_name,test_examples):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf.random.set_random_seed(2001)
    data = du.Data(jfilepath='config/config.json')
    data.set_batch_size( FLAGS.batch_size)
    data.set_no_classes( FLAGS.no_classes)
    test_out_file = FLAGS.train_dir + '/' + model_name + '_df.csv'
     
    #training dataset
    data.set_data_file('test')
    _test_dataset, _test_iterator = data.get_iterator2()
    test_init_op = _test_iterator.make_initializer(_test_dataset)
     
    tower_grads = []
    tower_ids = []
    tower_labels = []
    tower_probs = []
    tower_accu = []
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            #validation/Test set 
            loss_type = 'test_losses'
            test_image_ids, test_image_batch, test_label_batch = _test_iterator.get_next()
            test_accu, test_probs = tower_loss(scope, test_image_batch, test_label_batch, loss_type, test_image_ids)
            tower_ids.append( test_image_ids)
            tower_labels.append( test_label_batch)
            tower_probs.append( test_probs)
            tower_accu.append( test_accu)
    # Build test op for key variable which needs to be used to execute graph
    ids = tf.stack( tower_ids) 
    _probs = tf.stack( tower_probs) 
    labels = tf.stack( tower_labels) 
    labels = tf.reshape( labels, [-1]) 
    probs = tf.stack( tower_probs) 
    probs = tf.reshape( probs, [-1]) 
    accu = tf.reduce_mean(tower_accu)
     
    # Build an initialization operation to run below.
    local_var_init = tf.local_variables_initializer()
    #init = tf.global_variables_initializer()
     
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
     
    #saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
     
    saver.restore(sess, FLAGS.checkpoint_dir + '/' + model_name)
    #print("*****ckpt[%s] global_step[%s]" % (ckpt.model_checkpoint_path,global_step))
     
    sess.run(local_var_init)
    sess.run(test_init_op)
     
    a_image_ids = []
    a_preds = []
    a_probs = []
    a_labels = []
     
    for step in xrange(test_examples):
      start_time = time.time()
      test_ids, test_accu_value, test_label_value, test_probs_value, _test_probs_value = sess.run([ids, accu, labels, probs, _probs])
      duration = time.time() - start_time
      a_probs.extend(test_probs_value)
      a_image_ids.extend(test_ids)
      #a_preds.append(test_preds_value)
      a_labels.extend(test_label_value)
       
      #save results as binary image 
      data.save_results(test_ids,_test_probs_value) 
       
      #caculate accuracy with new methods. 
      pt50 = test_probs_value 
      pt50[pt50 >= .50 ] = 1.
      pt50[pt50 < .50 ] = 0.
      accu50 = 1 - np.abs(pt50 - test_label_value).mean()
       
      #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
       
      if step % 10 == 0:
        format_str = ('%s: step %d, '
                      ' accu50[%.5f] accu[%.5f]')
        print (format_str % (datetime.now(), step,
                              accu50,test_accu_value))
       
    pred_df = pd.DataFrame(list(zip(a_labels, a_probs)), 
               columns =['label', 'prob']) 
    pred_df.to_csv( test_out_file, index=False)
         
    #o_fd.close()
    #print_groups(test_out_file)
  

def test(model_name,test_examples):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf.random.set_random_seed(2001)
    data = du.Data(jfilepath='config/config.json')
    data.set_batch_size( FLAGS.batch_size)
    data.set_no_classes( FLAGS.no_classes)
    test_out_file = FLAGS.train_dir + '/' + model_name + '_df.csv'
     
    #training dataset
    data.set_data_file('test')
    _test_dataset, _test_iterator = data.get_iterator2()
    test_init_op = _test_iterator.make_initializer(_test_dataset)
     
    tower_grads = []
    tower_labels = []
    tower_probs = []
    tower_accu = []
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
             
            #validation/Test set 
            loss_type = 'test_losses'
            test_image_ids, test_image_batch, test_label_batch = _test_iterator.get_next()
            test_accu, test_probs = tower_loss(scope, test_image_batch, test_label_batch, loss_type, test_image_ids)
            tower_labels.append( test_label_batch)
            tower_probs.append( test_probs)
            tower_accu.append( test_accu)
    # Build test op for key variable which needs to be used to execute graph
    labels = tf.stack( tower_labels) 
    labels = tf.reshape( labels, [-1]) 
    probs = tf.stack( tower_probs) 
    probs = tf.reshape( probs, [-1]) 
    accu = tf.reduce_mean(tower_accu)
     
    #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    # Build the summary operation based on the TF collection of Summaries.
    #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    summaries.append(tf.summary.scalar('test_accu', test_accu))
    #summaries.append(tf.summary.scalar('test_loss', test_loss))

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()
    summary_op = tf.summary.merge(summaries)
     
    # Build an initialization operation to run below.
    local_var_init = tf.local_variables_initializer()
    #init = tf.global_variables_initializer()
     
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
     
    #variable_averages = tf.train.ExponentialMovingAverage( cifar10.MOVING_AVERAGE_DECAY)
    #variables_to_restore = variable_averages.variables_to_restore()
     
    #saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
     
    saver.restore(sess, FLAGS.checkpoint_dir + '/' + model_name)
    #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #print("*****ckpt[%s] global_step[%s]" % (ckpt.model_checkpoint_path,global_step))
     
    ''' 
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print("*****ckpt[%s] global_step[%s]" % (ckpt.model_checkpoint_path,global_step))
    else:
      print('No checkpoint file found')
      return
    ''' 
    #sess.run(init)
    sess.run(local_var_init)
    #sess.run(training_init_op)
    sess.run(test_init_op)
     
    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)

    a_image_ids = []
    a_preds = []
    a_probs = []
    a_labels = []
     
    for step in xrange(test_examples):
      start_time = time.time()
      #test_accu_value, test_image_ids_value, test_label_value, test_probs_value = sess.run([test_accu, test_image_ids, test_label_batch, test_probs])
      test_accu_value, test_label_value, test_probs_value = sess.run([accu, labels, probs])
      #a_image_ids.append(test_image_ids_value)
      #a_preds.append(test_preds_value)
      a_labels.extend(test_label_value)
      a_probs.extend(test_probs_value)
      duration = time.time() - start_time
       
      #save results as binary image 
      #data.save_results(test_image_ids_value,test_probs_value) 
       
      #caculate accuracy with new methods. 
      pt50 = test_probs_value 
      pt50[pt50 >= .50 ] = 1.
      pt50[pt50 < .50 ] = 0.
      accu50 = 1 - np.abs(pt50 - test_label_value).mean()
       
      #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
       
      if step % 10 == 0:
        format_str = ('%s: step %d, '
                      ' accu50[%.4f] accu[%.4f]')
        print (format_str % (datetime.now(), step,
                              accu50,test_accu_value))
       
      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
    '''
    o_fd = open( test_out_file, 'w')
    o_fd.write("%s,%s,%s,%s\n" % ('img_id','label','pred','prob'))
    img_width = data.get_img_width()
    img_heigth = data.get_img_heigth()
    img_dir_path = data.get_img_dir_path()
    img_filename_ext = data.get_img_filename_ext()
    for i,ids in enumerate(a_image_ids):
      for j,id in enumerate(ids):
        #o_fd.write("%s,%d,%d,%.5f\n" % (a_image_ids[i][j],np.argmax(a_labels[i][j]),a_preds[i][j],a_probs[i][j]))
        #print("%s %s" % (a_image_ids[i][j],a_probs[i][j][:10]))
        imgpath = img_dir_path + str(a_image_ids[i][j],"utf-8") + '_pi' + img_filename_ext #recreate original file URI
          
        img = np.reshape(a_probs[i][j],(img_width,img_heigth)) #recreate binary image of original size from flatten array
        np.save(imgpath,img) #save predicted results as binary image
    '''
    pred_df = pd.DataFrame(list(zip(a_labels, a_probs)), 
               columns =['label', 'prob']) 
    pred_df.to_csv( test_out_file, index=False)
         
    #o_fd.close()
    #print_groups(test_out_file)
  
def print_groups(i_file,g_count_key='prob',g_keys=['label','pred'],g_sort_keys=['label','pred']):
  fname = 'print_groups'
   
  df = pd.read_csv( i_file)
  #df.intent = df.intent.fillna('NA')
  g_df = df[ \
      #(df.status_code != 200) & \
      (df[g_count_key] > 0) \
          ] \
      .groupby(g_keys) \
      [g_count_key].count() \
      .nlargest(1000) \
      .reset_index(name='count') \
      .sort_values( g_sort_keys, ascending=True)
  print(g_df)

def main(argv=None):  # pylint: disable=unused-argument
  mode = 'test'
  steps = '499'
   
  #model_name = 'resnet_basic_lr01.cpkt'
  #model_name = 'incep_v3_he_da_wce10.cpkt'
  model_name = 'incepv3_kaggle_pretrain_lr01.cpkt'
   
  #cifar10.maybe_download_and_extract()
  if len(argv) > 0:
    print("arguments passed",argv[:])
    print("Running mode - [%s]" % argv[1])
    mode = argv[1]
    steps = argv[2]
  #print_groups(i_file='/tmp/cifar10_train/test_out_df.csv')
  #'''
   
  if mode == 'train':
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    train(model_name)
    #test(model_name,test_examples=100)
  else:
    model_name = model_name + '-' + steps
    test2(model_name,test_examples=100)

if __name__ == '__main__':
  tf.app.run()
