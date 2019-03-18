import os

import numpy as np
import pandas as pd
import sys
import os
import psutil

import tensorflow as tf
import keras

sys.path.append('../')

from utils import config as cutil
from utils import json_util as jutil
from utils import myImg2 as myimg

#tf.enable_eager_execution()

class Data(object):
  def log(self, mname, msg, level=0):
    sep = '| '
    if level <= self.verbose:
      print("##" + "Data::" + mname + sep + msg)
   
  def __init__(self, id="rn1_",config=None,jfilepath="../config/config.json"):
    mname = "__init__"
    
    self.id = id #shape of largest image unles restricted.
    self.img_buf_size = None #shape of largest image unles restricted.
    self.proc = psutil.Process()
     
    #Load / Initialize config source 
    if config == None:
      self.config = jutil.JsonUtil(jfilepath)
    else:
      self.config = config
     
    self.verbose = int(self.config.getElementValue(elem_path='/common/verbose'))
    #print("verbose[{}]".format(self.verbose)) 
    self.log( mname, "Initialized verbose[{}]".format(self.verbose), level=3)
    self.initialize_from_config()
   
  def initialize_from_config(self):
    mname = "initialize_from_config"

    self.no_classes = 5     
    self.pre_fetch = 1     
    self.batch_random_seed = 1001     
    self.cur_batch_offset = 0 
    self.processing_cnt = 0 
    self.img_buf_size = None #shape of largest image unles restricted.
    self.cdir = self.config.getElementValue(elem_path='/common/cdir')
    self.train_data_dir = self.config.getElementValue(elem_path='/common/data_dir_path')
    self.train_label_data_file = self.train_data_dir + self.config.getElementValue(elem_path='/train/label_data_file')
    self.log( mname, "Reading train_label_data_file[{}]".format(self.train_label_data_file), level=3)
    self.training_dataset_ratio = self.config.getElementValue(elem_path='/model/param/training_dataset_ratio')
    self.log( mname, "training_dataset_ratio[{}]".format(self.training_dataset_ratio), level=3)
    self.validation_dataset_ratio = self.config.getElementValue(elem_path='/model/param/validation_dataset_ratio')
    self.log( mname, "validation_dataset_ratio[{}]".format(self.validation_dataset_ratio), level=3)
    self.random_seed = self.config.getElementValue(elem_path='/model/param/random_seed')
    self.log( mname, "random_seed[{}]".format(self.random_seed), level=3)
     
    self.img_dir_path = self.config.getElementValue(elem_path='/img/img_dir_path')
    self.img_croped_dir_path = self.config.getElementValue(elem_path='/img/img_croped_dir_path')
    self.img_filename_ext = self.config.getElementValue(elem_path='/img/img_filename_ext')
    self.img_width = self.config.getElementValue(elem_path='/img/img_width')
    self.img_heigth = self.config.getElementValue(elem_path='/img/img_heigth')
     
    self.log( mname, "Images will be read from [{}]".format(self.img_dir_path), level=3)
    self.log( mname, "Image file extension [{}]".format(self.img_filename_ext), level=3)
    self.log( mname, "Image width [{}] heigth [{}]".format(self.img_width,self.img_heigth), level=3)
     
    self.img_processing_capacity = self.config.getElementValue(elem_path='/img/img_processing_capacity')
    self.log( mname, "img_processing_capacity[{}]".format(self.img_processing_capacity), level=3)
     
    self.batch_size = self.config.getElementValue(elem_path='/model/param/batch_size')
    self.log( mname, "batch_size[{}]".format(self.batch_size), level=3)
     
    self.channels = self.config.getElementValue(elem_path='/img/channels')
    self.log( mname, "channels[{}]".format(self.channels), level=3)
     
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
   
  def load_train_data(self):
    mname = "load_train_data"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_train = np.zeros(( 0, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h, 3), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    for i,rec in self.df.iterrows():
      #if cnt > 50:
      #  break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write("%6d out of %6d" % (cnt,tot_cnt))
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      #skip already processed data 
      if os.path.exists(self.img_croped_dir_path + rec.image + self.img_filename_ext):
        cnt += 1
        continue
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=imgpath) 
         
        i_w, i_h = myimg1.getImageDim() 
        croped_img_arr = np.zeros((n_img_w,n_img_h,3),dtype='uint8') 
        calc_img_w_offset = int((n_img_w - i_w)/2)
        calc_img_h_offset = int((n_img_h - i_h)/2)
        croped_img_arr[ calc_img_w_offset:(calc_img_w_offset + i_w), calc_img_h_offset:(calc_img_h_offset + i_h), :] = myimg1.getImage()
         
        ''' 
        croped_img = tf.image.resize_image_with_crop_or_pad( myimg1.getImage(), n_img_w, n_img_h)
        init = tf.global_variables_initializer()
        croped_img_arr = 0
        with tf.Session() as sess:
          sess.run(init)
          croped_img_arr = sess.run(croped_img)
          print(v.shape,type(v))  # will show you your variable.
          v = np.reshape( v, ( n_img_w, n_img_h, 3))
          print(v.shape,type(v))  # will show you your variable.
        ''' 
         
        x_img_buf[ 0, :, :, :] = croped_img_arr
         
        #'''#use below block of code to debug croped image with original.
        #myimg1.showImage()
        #myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=False)
        myimg2 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=rec.image+self.img_filename_ext, img=croped_img_arr) 
        myimg2.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
        #myimg2.saveImage()
        #'''
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
         
        #x_train = np.vstack( (x_train, x_img_buf))
        #x_train[cnt,:,:,:] = croped_img_arr
        y_buf.append(rec.level)
         
        self.df.loc[i,'imgexists'] = True
        self.df.loc[i,'w'], self.df.loc[i,'h'] = myimg1.getImageDim()
        self.df.loc[i,'_w'], self.df.loc[i,'_h'] = croped_img_arr.shape[0],croped_img_arr.shape[1]
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      cnt += 1
      
    #create y array as required
    y_train = np.array( y_buf, dtype='uint8')
    y_train = np.reshape( y_train, (y_train.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_train [{}] y_train [{}]".format(x_train.shape,y_train.shape), level=3)
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    self.df.to_csv( self.train_data_dir + 'u_img_set.csv')
     
   
  def load_data_as_greyscale(self):
    mname = "load_data_as_greyscale"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
    self.log( mname, "Loaded [{}] recs".format(self.df['level'].count()), level=3)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_train = np.zeros(( 0, n_img_w, n_img_h), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    for i,rec in self.df.iterrows():
      #if cnt >= 50:
      #  break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write("%6d out of %6d" % (cnt,tot_cnt))
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      #skip already processed data 
      if os.path.exists(self.img_croped_dir_path + rec.image + self.img_filename_ext):
        cnt += 1
        continue
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=rec.image, config=self.myImg_config, path=imgpath) 
         
        myimg1.getGreyScaleImage2(convertFlag=True) 
        myimg1.padImage(n_img_w,n_img_h)
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
         
        myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
         
        #x_train = np.vstack( (x_train, x_img_buf))
        #x_train[cnt,:,:,:] = croped_img_arr
        y_buf.append(rec.level)
         
        self.df.loc[i,'imgexists'] = True
        self.df.loc[i,'w'], self.df.loc[i,'h'] = myimg1.getImageDim()
        #self.df.loc[i,'_w'], self.df.loc[i,'_h'] = croped_img_arr.shape[0],croped_img_arr.shape[1]
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      cnt += 1
      
    #create y array as required
    y_train = np.array( y_buf, dtype='uint8')
    y_train = np.reshape( y_train, (y_train.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_train [{}] y_train [{}]".format(x_train.shape,y_train.shape), level=3)
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    self.df.to_csv( self.train_data_dir + 'u_img_set.csv')
   
  def load_img_data(self):
    mname = "load_greyscale_data"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
    self.log( mname, "Loaded [{}] recs".format(self.df['level'].count()), level=3)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.img_processing_capacity 
    if tot_cnt == 0:
      tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #generate dataset for handling train : test
    np.random.seed(self.random_seed)
    train_dataset_sample = np.random.choice( range(0,tot_cnt), int(tot_cnt * self.training_dataset_ratio), replace=False) 
    train_dataset_indicies = dict(zip(train_dataset_sample,train_dataset_sample))
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    y_train_buf = []
    y_test_buf = []
     
    if self.channels == 1:
      x_train = np.zeros(( len(train_dataset_sample), n_img_w, n_img_h), dtype='uint8')
      x_test = np.zeros(( (tot_cnt-len(train_dataset_sample)), n_img_w, n_img_h), dtype='uint8')
    else:
      x_train = np.zeros(( len(train_dataset_sample), n_img_w, n_img_h, self.channels), dtype='uint8')
      x_test = np.zeros(( (tot_cnt-len(train_dataset_sample)), n_img_w, n_img_h, self.channels), dtype='uint8')
     
    y_train = np.zeros((0,1),dtype='uint8')
    y_test = np.zeros((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    train_cnt = 0
    test_cnt = 0
    train_samples_cnt = len(train_dataset_sample)
    test_samples_cnt = tot_cnt - len(train_dataset_sample)
    self.log( mname, "[{}] recs for training.".format(train_samples_cnt), level=3)
    self.log( mname, "[{}] recs for test.".format(test_samples_cnt), level=3)
     
    for i,rec in self.df.iterrows():
      if cnt >= tot_cnt:
        break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write(progress_sts)
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=rec.image, config=self.myImg_config, path=imgpath) 
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
        if train_dataset_indicies.get(cnt,False): 
          #x_train = np.vstack( (x_train, x_img_buf))
          if train_cnt < train_samples_cnt:
            if self.channels == 1:
              x_train[train_cnt,:,:] = myimg1.getImage()
            else:
              x_train[train_cnt,:,:,:] = myimg1.getImage()
          y_train_buf.append(rec.level)
          train_cnt += 1
        else:
          #x_test = np.vstack( (x_test, x_img_buf))
          #self.log( mname, "[{}] [{}] x_test[{}] x_img_buf[{}]".format(cnt,test_cnt,x_test.shape,x_img_buf.shape), level=2)
          if test_cnt < test_samples_cnt:
            if self.channels == 1:
              x_test[test_cnt,:,:] = myimg1.getImage()
            else:
              x_test[test_cnt,:,:,:] = myimg1.getImage()
          y_test_buf.append(rec.level)
          test_cnt += 1
         
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      cnt += 1
      
    #create y array as required
    y_train = np.array( y_train_buf, dtype='uint8')
    y_train = np.reshape( y_train, (y_train.size,1))
    y_test = np.array( y_test_buf, dtype='uint8')
    y_test = np.reshape( y_test, (y_test.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_train [{}] y_train [{}]".format(x_train.shape,y_train.shape), level=3)
    self.log( mname, "x_test [{}] y_test [{}]".format(x_test.shape,y_test.shape), level=3)
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    #self.df.to_csv( self.train_data_dir + 'u_img_set.csv')
    
    return (x_train, y_train), (x_test, y_test)
   
  def initialize_for_batch_load(self):
    mname = "initiliaze_for_batch_load"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
    self.log( mname, "Loaded [{}] recs".format(self.df['level'].count()), level=3)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
    
    ''' 
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
    ''' 
     
    #generate dataset for handling train : test
    np.random.seed(self.random_seed)
    self.train_df = self.df.sample( frac=.7, replace=False, random_state=self.random_seed)
    self.test_df = self.df[~self.df.index.isin(self.train_df.index)]
    
    self.tot_cnt = self.img_processing_capacity 
    if self.tot_cnt == 0:
      self.tot_cnt = self.train_df['level'].count()
    cnt = 0
    file_missing = 0
     
    #self.train_df = self.train_df.reset_index()
    #self.test_df = self.test_df.reset_index()
    print(self.train_df.head())
     
    self.train_df.to_csv( self.train_data_dir + "train_df.csv", index=False, header=False)
    self.test_df.to_csv( self.train_data_dir + "test_df.csv", index=False, header=False)
    
    self.no_classes = self.train_df.level.nunique()
     
    return (self.train_df.level.count(), self.test_df.level.count()) 
  
  def get_batch_size(self):
    return self.batch_size
  
  def set_no_classes(self,no_classes):
    self.no_classes = no_classes
     
    return True
  
  def set_batch_size(self,batch_size):
    self.batch_size = batch_size
     
    return True
  
  def get_input_shape(self):
    n_img_w = self.img_width
    n_img_h = self.img_heigth
    
    '''
    if self.channels == 1:
      return np.zeros(( n_img_w, n_img_h, 1), dtype='uint8').shape
    else:
      return np.zeros(( n_img_w, n_img_h, self.channels), dtype='uint8').shape
    '''
    return np.zeros(( n_img_w, n_img_h, self.channels), dtype='uint8').shape
   
  def image_data_generator(self,mode="train"):
    #initialize all variables... 
    mname = 'image_data_generator'
    
    fd = open( self.train_data_dir + mode + '_df.csv', 'r')
    
    while True:
      n_img_w = self.img_width
      n_img_h = self.img_heigth
       
      #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
      x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
      x_buf = None
      y_buf = None
      y_labels = []
       
      img_cnt = self.batch_size
      '''
      if self.channels == 1:
        x_buf = np.zeros((img_cnt, n_img_w, n_img_h), dtype='uint8')
      else:
        x_buf = np.zeros((img_cnt, n_img_w, n_img_h, self.channels), dtype='uint8')
      '''
      x_buf = np.zeros((img_cnt, n_img_w, n_img_h, self.channels), dtype='uint8')
       
      y_buf = np.zeros((0,1),dtype='uint8')
       
      #loop in through dataframe. 
      #self.log( mname, "[{}] recs for set.".format(img_cnt), level=3)
       
      cnt = 0
      file_missing = 0
      while cnt < img_cnt:
        #if cnt >= tot_cnt:
        #  break
        
        line = fd.readline()
        if line == "":
           fd.seek(0)
           line = fd.readline()
        line = line.strip().split(',')
        image_id = line[0] 
        label = line[1] 
         
        imgpath = self.img_dir_path + image_id + self.img_filename_ext 
  
        '''       
        progress_sts = "[%.3fG] [%6d] %6d out of %6d [%45s]" % (self.proc.memory_full_info()[1]/(1024**3),self.processing_cnt,cnt,img_cnt,imgpath)
        sys.stdout.write(progress_sts)
        sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
        sys.stdout.flush()
        '''       
         
        if os.path.exists(imgpath):
          myimg1 = myimg.myImg( imageid=image_id, config=self.myImg_config, path=imgpath) 
          #myimg1.getGreyScaleImage2(convertFlag=True)
          #self.channels = 1
          #myimg1.padImage(n_img_w,n_img_h)
           
          #x_img_buf[ 0, :, :] = myimg1.getImage()
          if self.channels == 1:
            x_buf[cnt,:,:,0] = myimg1.getImage()
          else:
            x_buf[cnt,:,:,:] = myimg1.getImage()
           
          y_labels.append(label)
          #x_test = np.vstack( (x_test, x_img_buf))
          #self.log( mname, "[{}] [{}] x_test[{}] x_img_buf[{}]".format(cnt,test_cnt,x_test.shape,x_img_buf.shape), level=2)
           
          #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
          cnt += 1
          self.processing_cnt += 1
        else:
          print( mname, "****Image file [{}] doesn't exists!!!".format(imgpath))
          self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
          file_missing += 1
         
      #create y array as required
      y_buf = np.array( y_labels, dtype='uint8')
      y_buf = np.reshape( y_buf, (y_buf.size,1))
      #print final dimensionf or x_train and y_train
      #self.log( mname, "x_buf [{}] y_buf [{}]".format(x_buf.shape,y_buf.shape), level=3)
      #print( mname, "####x_test [{}] y_test [{}] y_buf[{}]".format(x_test.shape,y_test.shape,len(y_test_buf)))
        
      #self.log( mname, "Process dataset [{}]".format(cnt), level=3)
      #self.log( mname, "File missing [{}]".format(file_missing), level=3)
      #self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
      #print(self.df.head(10))
      
      # Normalize data.
      x_buf = x_buf.astype('float32') / 255
      
      # Convert class vectors to binary class matrices.
      #y_buf = keras.utils.to_categorical(y_buf, self.no_classes)
      y_buf = y_buf.astype('uint8')

      yield (x_buf, y_buf)
     
   
  def get_test_data(self):
    #initialize all variables... 
    mname = 'get_test_data'
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
    x_test = None
    y_test = None
    y_test_buf = []
     
    test_cnt = self.test_df.level.count()
    if self.channels == 1:
      x_test = np.zeros((test_cnt, n_img_w, n_img_h), dtype='uint8')
    else:
      x_test = np.zeros((test_cnt, n_img_w, n_img_h, self.channels), dtype='uint8')
     
    y_test = np.zeros((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    self.log( mname, "[{}] recs for testing.".format(test_cnt), level=3)
     
    cnt = 0
    file_missing = 0
    for i,rec in self.test_df.iterrows():
      #if cnt >= tot_cnt:
      #  break
       
      progress_sts = "%6d out of %6d" % (cnt,test_cnt)
      sys.stdout.write(progress_sts)
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.test_df.loc[i,'imgpath'] = imgpath
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=rec.image, config=self.myImg_config, path=imgpath) 
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
        if self.channels == 1:
          x_test[test_cnt,:,:] = myimg1.getImage()
        else:
          x_test[test_cnt,:,:,:] = myimg1.getImage()
         
        y_test_buf.append(rec.level)
        #x_test = np.vstack( (x_test, x_img_buf))
        #self.log( mname, "[{}] [{}] x_test[{}] x_img_buf[{}]".format(cnt,test_cnt,x_test.shape,x_img_buf.shape), level=2)
         
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      self.processing_cnt += 1
      cnt += 1
      
    #create y array as required
    y_test = np.array( y_test_buf, dtype='uint8')
    y_test = np.reshape( y_test, (y_test.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_test [{}] y_test [{}]".format(x_test.shape,y_test.shape), level=3)
    #print( mname, "####x_test [{}] y_test [{}] y_buf[{}]".format(x_test.shape,y_test.shape,len(y_test_buf)))
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    
    return (x_test, y_test)
     
  def image_generator(self,mode="train"):
    #initialize all variables... 
    mname = 'image_data_generator'
    
    fd = open( self.train_data_dir + mode + '_df.csv', 'r')
    
    while True:
      n_img_w = self.img_width
      n_img_h = self.img_heigth
      channels = self.channels
       
      #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
      #x_img_buf = np.empty(( n_img_w, n_img_h), dtype='uint8')
      x_buf = np.zeros(( n_img_w, n_img_h, channels), dtype='uint8')
      y_buf = np.zeros((1),dtype='uint8')
       
      #loop in through dataframe. 
      #self.log( mname, "[{}] recs for set.".format(img_cnt), level=3)
       
      cnt = 0
      file_missing = 0
      #while cnt < img_cnt:
        #if cnt >= tot_cnt:
        #  break
        
      line = fd.readline()
      if line == "":
         fd.seek(0)
         line = fd.readline()
      line = line.strip().split(',')
      image_id = line[0] 
      label = line[1] 
       
      imgpath = self.img_dir_path + image_id + self.img_filename_ext 

      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=image_id, config=self.myImg_config, path=imgpath) 
        #myimg1.getGreyScaleImage2(convertFlag=True)
        #self.channels = 1
        #myimg1.padImage(n_img_w,n_img_h)
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
        if self.channels == 1:
          x_buf[:,:,0] = myimg1.getImage()
        else:
          x_buf[:,:,:] = myimg1.getImage()
         
        #y_labels.append(label)
         
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        cnt += 1
        self.processing_cnt += 1
      else:
        print( mname, "****Image file [{}] doesn't exists!!!".format(imgpath))
        self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
         
      #create y array as required
      y_buf = np.array( label, dtype='uint8')
      y_buf = y_buf.astype('float32')
      y_buf = np.reshape( y_buf, (1))
      y_buf = keras.utils.to_categorical(y_buf, self.no_classes)
      #y_buf = np.reshape( y_buf, (self.no_classes,1))
      #print("XXXXX",image_id,label,y_buf,y_buf.shape)
       
      #x_buf = x_buf.astype('float32') / 255
      x_buf = x_buf.astype('float32')
      x_buf /= 255.0
      x_buf -= np.mean(x_buf)
      # Crop the central [height, width] of the image.
      #x_buf = tf.cast( x_buf, tf.float32)
      #x_buf = tf.image.resize_image_with_crop_or_pad(x_buf,n_img_h,n_img_w)
       
      # Subtract off the mean and divide by the variance of the pixels.
      #x_buf = tf.image.per_image_standardization(x_buf)
       
      # Set the shapes of tensors.
      #x_buf.set_shape([n_img_w, n_img_h, channels])
       
      ''' 
      m = re.findall('(^\d+)_(.*?)$',image_id)
      _id = tf.cast(m[0][0],tf.int64)
      ''' 
       
      yield (x_buf, y_buf)
      #yield (x_buf, y_buf, _id)
   
  def get_iterator(self):
     
    dataset = tf.data.Dataset.from_generator( \
                 self.image_generator, \
                 (tf.float32, tf.float32), \
                 #(tf.uint8, tf.float32, tf.int64), \
                 #(tf.TensorShape([self.img_width,self.img_heigth]),tf.TensorShape([1,5],tf.TensorShape[1])))
                 (tf.TensorShape([self.img_width,self.img_heigth,self.channels]),tf.TensorShape([1,self.no_classes])))
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.shuffle(buffer_size=self.pre_fetch*self.batch_size,seed=self.batch_random_seed)
    _iterator = dataset.make_initializable_iterator()
    #_iterator = dataset.make_one_shot_iterator()
     
    return dataset,_iterator
     
if __name__ == "__main__":
  #prep_data()
  data = Data()
  a1 = tf.Variable( 0, dtype='uint8')
  #data.load_data_as_greyscale()
  #data.load_img_data()
   
  #data.initialize_for_batch_load()
  #'''
  with tf.Graph().as_default(), tf.device('/cpu:0'):
  #with tf.Session() as sess:
      #iterator = data.get_iterator()
      #for X, Y in data.get_iterator():
      _dataset,_iterator = data.get_iterator()
      #training_init_op = _iterator.make_initializer(_dataset)
      #sess.run(training_init_op)
      #_iterator.initializer()
      X, Y = _iterator.get_next()
      #X, Y, ID = _iterator.get_next()
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      for i in range(3):
        #X, Y = _iterator.get_next()
        x, y = X, Y
        #x, y, _id = X, Y, ID
        #print("***",np.shape(x.eval()),np.shape(y.eval()),"****",y.eval())
        print("***",np.shape(x),x.get_shape().as_list()[1],np.shape(y),"****",y)
        #print("***",np.shape(x.eval()),np.shape(y.eval()),"****",_id.eval(),y.eval())
        #a1 += i
   
      #init_op = tf.global_variables_initializer()
      
      with tf.Session() as sess:
        sess.run(x,init_op)   
        #sess.run(training_init_op)   
        print("a1=%d" % (sess.run(a1)))
