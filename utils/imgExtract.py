
import os
import sys
import copy

import cv2

import pandas as pd
import numpy as np
import tensorflow as tf

sys.path.append('../')

from utils import config as cutil
from utils import myImg2 as myimg
from utils import cmdopt_util as cmd_util

class myImgExtractor:
  def __init__(self,id,imgdir,img_size,tdir,patch_size,patch_stride,truth_pixel=1):
    self.cn = 'myImgExtractor'
    fn = '__init__'
   
    self.id = id
    self.ddir = '../../data/'
    self.imgdir = imgdir
    self.tdir = tdir
    self.img_size = img_size
    self.img_config_fl = '../config/img_conf.csv' #file holding all marked masked/regions of images
    self.mask_df = None #DF to read image mask params
    
    #patch specific params
    self.truth_pixel = truth_pixel #set ideally to 1 or to 255 for viewing purpose.
    self.patch_size = patch_size
    self.patch_stride = patch_stride
     
    self.mylog(fn,"Invoked with id[%s] imgdir[%s] img_size[%d]" % (id,imgdir,img_size))
    
    #mandatory checks
    if os.path.isdir(self.imgdir) == False:
      #exit if image directory is not proper initialized
      self.mylog(fn,"imgdir[%s] argument to constructor not defined or invalid path" % (self.imgdir))
      sys.exit(-1)
     
    #mandatory checks
    if os.path.isdir(self.tdir) == False:
      #exit if image directory is not proper initialized
      self.mylog(fn,"tdir[%s] argument to constructor not defined or invalid path" % (self.tdir))
      sys.exit(-1)
    
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.imgdir)
    self.myImg_config.setDdir( self.imgdir)
    self.myImg_config.setOdir( self.imgdir)
    self.myImg_config.setIdir( self.imgdir)
       
    #read image config
    self.read_img_mask_config()
  
  def mylog(self,fn='',msg=''):
    print(self.cn,'[',self.id,']',fn,': ',msg)
  
  def read_img_mask_config( self, img_shape=[512,512,3]):
    fn = 'read_img_mask_config' #function name
    
    col_names = ['img_id','mask_type','x','y','w','h','mask_label']
    i_fl = self.img_config_fl
     
    img_w,img_h,img_channels = img_shape #input image shape
    img = np.zeros(img_shape,dtype='uint8')
     
    if i_fl is None: 
      self.mylog(fn,"Image mask file config path[%s] is Null." % (i_fl))
      sys.exit(-1)
    
    if os.path.exists(i_fl):
      self.mylog(fn,"Reading file[%s]" % (i_fl))
      self.mask_df = pd.read_csv( i_fl)
      self.mylog(fn,"Records loaded[%d]" % (self.mask_df['img_id'].count()))
      self.mylog(fn,"Target image shape - %s" % (img_shape))
    else:
      self.mylog(fn,"Image mask file config path[%s] doesnt exist." % (i_fl))
      sys.exit(-2)
      
    #Reset all column names for quick usage. 
    #filename,shape_attributes.name,shape_attributes.x,shape_attributes.y,shape_attributes.width,shape_attributes.height,region_attributes.Name
     
    #Sort file as per usage progression logic
    #self.mask_df.columns = col_names
    #self.mask_df = self.mask_df.sort_values(['img_id','x','y'],ascending=[True,True,True])
    #print(mask_df.head())
   
    ''' 
    #main df loop
    cnt = 0 #processing counter
    for i,rec in self.mask_df.iterrows():
      if cnt > 5:
        break
      self.mylog(fn,"%d %s %d %d %d %d %s" % (cnt,rec.img_id, rec.x, rec.y, rec.w, rec.h, rec.mask_label))
      
      cnt += 1
    ''' 
    
    #return sucess
    return True
  
  #function only to be called once mask_df is populated from config file.
  #function returns all availables masks for given image_id.
  def get_mask_for_image_id( self, img_id=None):
    fn = 'get_mask_for_image_id'
    if img_id is None:
      self.mylog(fn,"img_id param passed null.")
      sys.exit(-3)
    else:
      img_mask_df = self.mask_df[self.mask_df.img_id == img_id] #select records for spcific image/img_id
      self.mylog(fn,"[%d]recs selceted for img_id[%s]" % (img_mask_df['img_id'].count(),img_id))
     
    #return patch of DF for given img_id
    return img_mask_df
   
  ''' 
    This function just opens original image and its respect ground truth image for preparing training patches.
    Technically both images i.e. original and ground truth (target image) should always be of same size.
    the patch window for data prep will move simulataneously for both images to build pair patches. Every
    patch of original image will have respective pair of target image or ground truth. 
    Every pixel of ground truth is expcted result and every pixel will be predicted by model.
  ''' 
  def get_image_and_ground_truth( self, img_id):
     fn = 'get_image_and_ground_truth'
      
     #self.mylog(fn,"Generating ground truth...")
     
     img_path = self.imgdir + 'images/' + img_id + '.jpg'
     mask_path = self.imgdir + 'mask/' + img_id + '_mask.jpg'
     myimg1 = None
     #self.mylog(fn,"processing img_id[%s] img_path[%s]..." % (img_id,img_path))
      
     if os.path.exists(img_path):
       myimg1 = myimg.myImg( imageid=img_id, config=self.myImg_config, path=img_path) 
     else:
       print("Image file [%s] does not exists..." % (img_path))
       sys.exit(-1)
          
     if os.path.exists(mask_path):
       myimg2 = myimg.myImg( imageid=img_id, config=self.myImg_config, path=mask_path) 
     else:
       print("Image mask file [%s] does not exists..." % (mask_path))
       sys.exit(-1)
          
     image_org = myimg1.getImage()
     #print("**********",image_org.shape,type(image_org.shape),type(image_org))
     #self.mylog(fn,"org image shape [%d %d %d]" % (image_org.shape))
     
     #initialize truth image data
     truth_img = myimg2.getImage()
     #set truth_img with 1's and 0's based on pixel
     #truth_img[truth_img>0] = 1.
     #self.mylog(fn,"truth image shape [%d %d %d]" % (truth_img.shape))
     
     #return patch of DF for given img_id
     return image_org, truth_img
  
  ''' 
     build ground truth from mask dataset. The mask will be used to build target region of image i.e. 1's 
    or while rest all remaining area within image becomes 0's.
  ''' 
  def build_ground_truth_from_mask( self, img_id):
     fn = 'build_ground_truth_from_mask'
      
     #self.mylog(fn,"Generating ground truth...")
     
     img_path = self.imgdir + img_id
     myimg1 = None
     #self.mylog(fn,"processing img_id[%s] img_path[%s]..." % (img_id,img_path))
      
     if os.path.exists(img_path):
       myimg1 = myimg.myImg( imageid='t1', config=self.myImg_config, path=img_path) 
     else:
       print("Image file [%s] does not exists..." % (img_path))
       sys.exit(-1)
          
     image_org = myimg1.getImage()
     masked_img = copy.copy(image_org)
     #print("**********",image_org.shape,type(image_org.shape),type(image_org))
     #self.mylog(fn,"org image shape [%d %d %d]" % (image_org.shape))
     truth_img = np.zeros(image_org.shape,dtype='uint8')  #create 0's based truth
     #self.mylog(fn,"truth image shape [%d %d %d]" % (truth_img.shape))
     
     img_mask_df = self.get_mask_for_image_id(img_id=img_id)
     #loop image mask df loop
     cnt = 0 #processing counter
     _img_sum = 0
     for i,rec in img_mask_df.iterrows():
       #if cnt > 5:
       #  break
       #self.mylog(fn,"%d %s %d %d %d %d %s" % (cnt,rec.img_id, rec.x, rec.y, rec.w, rec.h, rec.mask_label))
       #truth_img[rec.x:(rec.x+rec.w),rec.y:(rec.y+rec.h),:] = self.truth_pixel
       truth_img[rec.y:(rec.y+rec.h),rec.x:(rec.x+rec.w),:] = self.truth_pixel
       #self.mylog(fn,"%d %s patch sum %.1f image sum %.1f accu sum %.1f " % 
       #                     (cnt,rec.img_id, rec.w*rec.h, truth_img.sum(),_img_sum))
       masked_img = cv2.rectangle(masked_img, (rec.x, rec.y), (rec.x+rec.w, rec.y+rec.h), (255,0,0),1)
        
       cnt += 1
       _img_sum += rec.w * rec.h * truth_img.shape[-1]
      
     #myimg1.padImage( _w, _h)  #trim truth as per reduced size
       
     #return patch of DF for given img_id
     return image_org,truth_img, masked_img
  
  # Function to extract patches using 'extract_image_patches'
  def img_to_patches(self,raw_input, _patch_size=(128, 128), _stride=100):
    with tf.variable_scope('im2_patches'):
      patches = tf.image.extract_image_patches(
        images=raw_input,
        ksizes=[1, _patch_size[0], _patch_size[1], 1],
        strides=[1, _stride, _stride, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
      )
      h = tf.shape(patches)[1]
      w = tf.shape(patches)[2]
      patches = tf.reshape(patches, (patches.shape[0], -1, _patch_size[0], _patch_size[1], 3))
       
    return patches, (h, w)
   
  # Function to reconstruct image
  def patches_to_img(self,update, _block_shape, _stride=100):
    with tf.variable_scope('patches2im'):
      _h = _block_shape[0]
      _w = _block_shape[1]
      bs = tf.shape(update)[0]  # batch size
      np = tf.shape(update)[1]  # number of patches
      ps_h = tf.shape(update)[2]  # patch height
      ps_w = tf.shape(update)[3]  # patch width
      col_ch = tf.shape(update)[4]  # Colour channel count
       
      wout = (_w - 1) * _stride + ps_w  # Recalculate output shape of "extract_image_patches" including padded pixels
      hout = (_h - 1) * _stride + ps_h  # Recalculate output shape of "extract_image_patches" including padded pixels
       
      x, y = tf.meshgrid(tf.range(ps_w), tf.range(ps_h))
      x = tf.reshape(x, (1, 1, ps_h, ps_w, 1, 1))
      y = tf.reshape(y, (1, 1, ps_h, ps_w, 1, 1))
      xstart, ystart = tf.meshgrid(tf.range(0, (wout - ps_w) + 1, _stride),
                                   tf.range(0, (hout - ps_h) + 1, _stride))
  
      bb = tf.zeros((1, np, ps_h, ps_w, col_ch, 1), dtype=tf.int32) + tf.reshape(tf.range(bs), (-1, 1, 1, 1, 1, 1))  #  batch indices
      yy = tf.zeros((bs, 1, 1, 1, col_ch, 1), dtype=tf.int32) + y + tf.reshape(ystart, (1, -1, 1, 1, 1, 1))  # y indices
      xx = tf.zeros((bs, 1, 1, 1, col_ch, 1), dtype=tf.int32) + x + tf.reshape(xstart, (1, -1, 1, 1, 1, 1))  # x indices
      cc = tf.zeros((bs, np, ps_h, ps_w, 1, 1), dtype=tf.int32) + tf.reshape(tf.range(col_ch), (1, 1, 1, 1, -1, 1))  # color indices
      dd = tf.zeros((bs, 1, ps_h, ps_w, col_ch, 1), dtype=tf.int32) + tf.reshape(tf.range(np), (1, -1, 1, 1, 1, 1))  # shift indices
  
      idx = tf.concat([bb, yy, xx, cc, dd], -1)
  
      stratified_img = tf.scatter_nd(idx, update, (bs, hout, wout, col_ch, np))
      stratified_img = tf.transpose(stratified_img, (0, 4, 1, 2, 3))
      stratified_img_count = tf.scatter_nd(idx, tf.ones_like(update), (bs, hout, wout, col_ch, np))
      stratified_img_count = tf.transpose(stratified_img_count, (0, 4, 1, 2, 3))
       
      with tf.variable_scope("consolidate"):
          sum_stratified_img = tf.reduce_sum(stratified_img, axis=1)
          stratified_img_count = tf.reduce_sum(stratified_img_count, axis=1)
          reconstructed_img = tf.divide(sum_stratified_img, stratified_img_count)
       
      return reconstructed_img, stratified_img
 
  ''' 
     generate_data1 to used when we have rectangular or polygonal mask dataset for the ROI (region of interest)
       code will use image_id available to locate all its masks. Available masks will be used to prepare a 
       binary target image and mask image copy is prepare for reference or isualization. In target image a
       all pixels for mask or ROI are made '1' while all rest of the image area pixels are made '0'.
       Once target and mask images are build a patching algorithm is used to create image patches.
        Each patch of given image is uniquely name using index and then further prepended with 2 characters 
       that identifies type of image i.e. ti for target image, oi for origimal etc. So for image id 12_left.jpg
       all its patches for target will have names like 12_left_1_ti.jpg, 12_left_2_ti.jpg, 12_left_3_ti.jpg etc.
  ''' 
  def generate_data1(self):
    fn = 'generate_data'
    out_fl = 'patches_df.csv'
    _batch_size = 1 #batch size to process images
     
    #open file for writing DF for patches
    patch_fd = open( self.tdir + out_fl, 'w')
     
    #get img_id's for which this process nees to be run...
    img_ids = self.mask_df.img_id.unique()
     
    cnt = 0 
    for img_id in img_ids:
      #if cnt > 2:
      #  break
      self.mylog(fn,"Generating img_id[%s]" % (img_id))
      oi, ti, mi = self.build_ground_truth_from_mask(img_id)
      #mi = self.reshape_img( self.img_size, self.img_size, mi)
      #cv2.imshow(' masked image [' + img_ids[6] + ']', mi.astype(np.float32)/255)
      ti_ep = self.get_img_patches(img=ti)
      oi_ep = self.get_img_patches(img=oi)
      mi_ep = self.get_img_patches(img=mi)
      #Show extracted patches images
      for i in range(oi_ep.shape[1]):
        oi_im = oi_ep[0, i, :, :, :]
        ti_im = ti_ep[0, i, :, :, 0]
        mi_im = mi_ep[0, i, :, :, :]
         
        #save all files.
        id = img_id.split('.')[0]
        #cv2.imwrite( self.tdir + id + '_mi_' + str(i) + '.jpeg', mi_im)
        #cv2.imwrite( self.tdir + id + '_oi_' + str(i) + '.jpeg', oi_im)
        np.save( self.tdir + id + '_' + str(i) + '_oi', oi_im)
        np.save( self.tdir + id + '_' + str(i) + '_mi', mi_im)
        np.save( self.tdir + id + '_' + str(i) + '_ti', ti_im)
        patch_fd.write(  id + '_' + str(i) + '\n')
        '''
        if ti_im.sum() > 0:
          self.mylog(fn,"patch - [%d] sum[%.1f]" % (i,ti_im.sum()/(3*self.truth_pixel)))
           
          #stack images side by side 
          img_hstack = np.hstack((oi_im, ti_im))
          img_hconcat = np.concatenate((oi_im, ti_im), axis=1)
  
          #cv2.imshow(' img_hstack patch - [' + str(i) + ']', img_hstack.astype(np.float32)/255)
          #cv2.waitKey(0)
          cv2.imshow(' img_hconcat patch - [' + str(i) + ']', img_hconcat.astype(np.float32)/255)
          cv2.waitKey(0)
        '''
       
      cnt += 1
       
    #self.process_img(img_ids[0])
    
  ''' 
  ''' 
  def generate_data2(self):
    fn = 'generate_data2'
    out_fl = 'patch_df.csv'
    _batch_size = 1 #batch size to process images
     
    #open file for writing DF for patches
    patch_fd = open( self.tdir + out_fl, 'w')
     
    #get img_id's for which this process nees to be run...
    img_ids = self.mask_df.img_id.unique()
     
    cnt = 0 
    for img_id in img_ids:
      #if cnt > 2:
      #  break
      self.mylog(fn,"Generating img_id[%s]" % (img_id))
      id = img_id.split('.')[0]
      oi, ti = self.get_image_and_ground_truth(id)
      #mi = self.reshape_img( self.img_size, self.img_size, mi)
      #cv2.imshow(' masked image [' + img_ids[6] + ']', mi.astype(np.float32)/255)
      ti_ep = self.get_img_patches(img=ti)
      oi_ep = self.get_img_patches(img=oi)
       
      #Show extracted patches images
      for i in range(oi_ep.shape[1]):
        oi_im = oi_ep[0, i, :, :, :]
        ti_im = ti_ep[0, i, :, :, 0]
         
        #save all files.
        #cv2.imwrite( self.tdir + id + '_mi_' + str(i) + '.jpeg', mi_im)
        #cv2.imwrite( self.tdir + id + '_oi_' + str(i) + '.jpeg', oi_im)
        np.save( self.tdir + id + '_' + str(i) + '_oi', oi_im)
        np.save( self.tdir + id + '_' + str(i) + '_ti', ti_im)
        patch_fd.write(  id + '_' + str(i) + '\n')
        '''
        if ti_im.sum() > 0:
          self.mylog(fn,"patch - [%d] sum[%.1f]" % (i,ti_im.sum()/(3*self.truth_pixel)))
           
          #stack images side by side 
          img_hstack = np.hstack((oi_im, ti_im))
          img_hconcat = np.concatenate((oi_im, ti_im), axis=1)
  
          #cv2.imshow(' img_hstack patch - [' + str(i) + ']', img_hstack.astype(np.float32)/255)
          #cv2.waitKey(0)
          cv2.imshow(' img_hconcat patch - [' + str(i) + ']', img_hconcat.astype(np.float32)/255)
          cv2.waitKey(0)
        '''
       
      cnt += 1
       
    #self.process_img(img_ids[0])
    
  def reshape_img( self, n_img_w, n_img_h, imgbuf):
    i_w = imgbuf.shape[0]
    i_h = imgbuf.shape[1]
    i_channels = imgbuf.shape[2]
    s_img_w_offset = 0
    s_img_h_offset = 0
    t_img_w_offset = 0
    t_img_h_offset = 0
    calc_img_h = i_h
    calc_img_w = i_w
     
    calc_img_w_offset = abs(int((n_img_w - i_w)/2))
    calc_img_h_offset = abs(int((n_img_h - i_h)/2))
     
    if i_w >= n_img_w:
       t_img_w_offset = 0
       s_img_w_offset = calc_img_w_offset
       calc_img_w = n_img_w
    else:
       t_img_w_offset = calc_img_w_offset
       s_img_w_offset = 0
       calc_img_w = i_w
     
    if i_h >= n_img_h:
       t_img_h_offset = 0
       s_img_h_offset = calc_img_h_offset
       calc_img_h = n_img_h
    else:
       t_img_h_offset = calc_img_h_offset
       s_img_h_offset = 0
       calc_img_h = i_h
 
    #print("*****i_w[{}] i_h[{}] s_img_w_offset[{}] s_img_h_offset[{}] calc_img_w[{}] calc_img_h[{}] t_img_w_offset[{}] t_img_h_offset[{}] ".format(i_w,i_h,s_img_w_offset,s_img_h_offset,calc_img_w,calc_img_h,t_img_w_offset,t_img_h_offset))
     
    if i_channels == 3:
      croped_img_arr = np.zeros((n_img_w,n_img_h,i_channels),dtype='uint8') 
      croped_img_arr[ t_img_w_offset:(t_img_w_offset + calc_img_w), t_img_h_offset:(t_img_h_offset + calc_img_h), :] = imgbuf[ s_img_w_offset:(s_img_w_offset + calc_img_w), s_img_h_offset:(s_img_h_offset + calc_img_h), :]
    else:
      croped_img_arr = np.zeros((n_img_w,n_img_h),dtype='uint8') 
      croped_img_arr[ t_img_w_offset:(t_img_w_offset + calc_img_w), t_img_h_offset:(t_img_h_offset + calc_img_h)] = imgbuf[ s_img_w_offset:(s_img_w_offset + calc_img_w), s_img_h_offset:(s_img_h_offset + calc_img_h)]
     
    return croped_img_arr
    
  def get_img_patches(self,img=None):
    fn = 'process_img'
    _w = self.img_size
    _h = self.img_size
     
    if img is None:
      print("Image array passed cannot be null...")
      return False
    else:   
      # load initial image
       
      img = self.reshape_img( _w, _h, img)
       
      # Add batch dimension
      image = np.expand_dims(img, axis=0)
  
      # set parameters
      patch_size = (self.patch_size, self.patch_size)
      stride = self.patch_stride
  
      input_img = tf.placeholder(dtype=tf.float32, shape=image.shape, name="input_img")
      # Extract patches using "extract_image_patches()"
      extracted_patches, block_shape = self.img_to_patches(input_img, _patch_size=patch_size, _stride=stride)
      #self.mylog(fn, "extracted image shape %s" % (extracted_patches.shape))
      # block_shape is the number of patches extracted in the x and in the y dimension
      # extracted_patches.shape = (1, block_shape[0] * block_shape[1], patch_size[0], patch_size[1], 3)
       
      #reconstructed_img, stratified_img = self.patches_to_img(extracted_patches, block_shape, stride)  # Reconstruct Image
       
      with tf.Session() as sess:
          #ep, bs, ri, si = sess.run([extracted_patches, block_shape, reconstructed_img, stratified_img], feed_dict={input_img: image})
          ep, bs = sess.run([extracted_patches, block_shape], feed_dict={input_img: image})
      #si = si.astype(np.int32)
       
      '''
      print('*****************************************************************') 
      print('original image ',img.shape)
      print('extracted patches ',ep.shape)
      #print('block shape ',bs.shape)
      #print('reconstructed image  ',ri.shape)
      #print('stratified image  ',si.shape)
      print('*****************************************************************') 
      
      # Show reconstructed image
      cv2.imshow('sd', ri[0, :, :, :].astype(np.float32) / 255)
      cv2.waitKey(0)
       
      # Show stratified images
      for i in range(si.shape[1]):
        #if i > 2:
        #  break
        im_1 = si[0, i, :, :, :]
        cv2.imshow('sd', im_1.astype(np.float32)/255)
        cv2.waitKey(0)
       
      # Show extracted patches images
      for i in range(ep.shape[1]):
        #if i > 2:
        #  break
        im_1 = ep[0, i, :, :, :]
        cv2.imshow('sd', im_1.astype(np.float32)/255)
        cv2.waitKey(0)
      ''' 
      
      return ep
   
  def process_img_by_id(self,img_id=None):
    fn = 'process_img'
    _w = self.img_size
    _h = self.img_size
    myimg1 = None
     
    if img_id is None:
      print("Image path cannot be null...")
      return False
    else:   
      # load initial image
       
      img_path = self.imgdir + img_id 
      if os.path.exists(img_path):
        myimg1 = myimg.myImg( imageid='t1', config=self.myImg_config, path=img_path) 
      else:
        print("Image file [%s] does not exists..." % (imgpath))
        sys.exit(-1)
           
      i_w, i_h = myimg1.getImageDim() 
      #croped_img_arr = np.zeros((n_img_w,n_img_h,3),dtype='uint8') 
      image_org = myimg1.getImage()
      print('img_path ',img_path,' img shape ',image_org.shape)
      myimg1.padImage( _w, _h)
      image_org = myimg1.getImage()
      print('img_path ',img_path,' img shape ',image_org.shape)
       
      # Add batch dimension
      image = np.expand_dims(image_org, axis=0)
  
      # set parameters
      patch_size = (256, 256)
      stride = 128
  
      input_img = tf.placeholder(dtype=tf.float32, shape=image.shape, name="input_img")
      # Extract patches using "extract_image_patches()"
      extracted_patches, block_shape = self.img_to_patches(input_img, _patch_size=patch_size, _stride=stride)
      self.mylog(fn, "extracted image shape %s" % (extracted_patches.shape))
      # block_shape is the number of patches extracted in the x and in the y dimension
      # extracted_patches.shape = (1, block_shape[0] * block_shape[1], patch_size[0], patch_size[1], 3)
       
      #reconstructed_img, stratified_img = self.patches_to_img(extracted_patches, block_shape, stride)  # Reconstruct Image
      print('img shape ',image_org.shape)
       
      with tf.Session() as sess:
          #ep, bs, ri, si = sess.run([extracted_patches, block_shape, reconstructed_img, stratified_img], feed_dict={input_img: image})
          ep, bs = sess.run([extracted_patches, block_shape], feed_dict={input_img: image})
          # print(bs)
      #si = si.astype(np.int32)
       
      print('*****************************************************************') 
      print('original image ',image_org.shape)
      print('extracted patches ',ep.shape)
      #print('block shape ',bs.shape)
      #print('reconstructed image  ',ri.shape)
      #print('stratified image  ',si.shape)
      print('*****************************************************************') 
      
      '''
      # Show reconstructed image
      cv2.imshow('sd', ri[0, :, :, :].astype(np.float32) / 255)
      cv2.waitKey(0)
       
      # Show stratified images
      for i in range(si.shape[1]):
        #if i > 2:
        #  break
        im_1 = si[0, i, :, :, :]
        cv2.imshow('sd', im_1.astype(np.float32)/255)
        cv2.waitKey(0)
       
      # Show extracted patches images
      for i in range(ep.shape[1]):
        #if i > 2:
        #  break
        im_1 = ep[0, i, :, :, :]
        cv2.imshow('sd', im_1.astype(np.float32)/255)
        cv2.waitKey(0)
      ''' 

if __name__ == "__main__":
    img_path = cmd_util.get_imgpath(sys.argv[1:])
  #def __init__(self,id,imgdir,img_size,tdir,patch_size,patch_stride,truth_pixel=1):
    img_extractor = myImgExtractor(id='ie23',
                                     imgdir='/disk1/data1/data/idrid/ex/',
                                     img_size=2048,
                                     tdir='/disk1/data1/data/px_he/',
                                     patch_size=128,
                                     patch_stride=32,
                                     truth_pixel=1
                                    )
    img_extractor.generate_data2()
    #img_extractor.build_ground_truth('82_left.jpeg')
    #img_extractor.gen_images()
    #img_extractor.process_img(img_path=img_path)
    #img_extractor.read_img_mask_config(i_fl=img_path)
