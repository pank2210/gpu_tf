
import sys, getopt

import numpy as np
#from scipy import misc
import cv2 as misc
import math
import re
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#from matplotlib import pyplot as plt

import config as cutil

class myImg(object):
   cname = 'myImg'

   def __init__(self,path,imageid="x123",config=None,ekey="",img=None,channels=3):
      mname = '__init__' 
      
      self.id = imageid
      self.i_img_file_name = path

      if config is None:
        raise Exception("class[" + self.__class__.__name__ + "] error. Exception as config passed is Null.")
      else:
        self.config = config
        self.logger = self.config.logger
      
      self.ekey = ekey
      self.key = self.__class__.__name__
     
      self.logger.log( "myImg instance initialization. id[{}] key[{}] ekey[{}] imgpath[{}]".format(self.id,self.key,self.ekey,self.config.idir + path))
     
      #img param which is numpy image array takes precedence over path 
      #If path is passed along with img then path will be overwritten with img content as .jpg
      if type(img).__name__ != self.config.typeNone: #img is passed so use it.
         #print("##generating image from image array...")
         self.img = img
         if path is None: #construct if path doesn't exists when img is passed.
            self.imgpath = self.config.odir + 'img_' + self.id + '.jpg'
         else:
            #self.imgpath = self.config.odir + path
            if os.path.isfile(path):
              self.imgpath = path
            else:
              self.imgpath = self.config.odir + path
         #misc.imwrite( self.imgpath, img) #since img is passed, persist it in config.odir
      else: #path is passed, so use it to build img
         #check if mandatory param are passed. 
         if type(path).__name__ == self.config.typeNone:
            raise Exception("class[" + self.__class__.__name__ + "] error. A path<valid image path> argument cannot be null if img is None.")
         #print("reading from path[{}] [{}]".format(path,os.path.exists(path)))
         if os.path.exists(path):
           self.imgpath = path
         else:
           self.imgpath = self.config.idir + path
         #print("reading imgpath[{}]".format(self.imgpath))
         if channels == 3:
           self.img = misc.imread(self.imgpath) #1 param is for color/multi channel image read.
         else:
           self.img = misc.imread(self.imgpath,misc.IMREAD_GRAYSCALE) #1 param is for color/multi channel image read.
         #print("Read from path[{}] shape[{}]".format(path,self.img.shape))

      self.setImageMetadata()      
      self.imgdict = {} #initialize image dictionay
      #self.logger.log( "myImage instance initialization. id[{}] configid[{}] imgpath[{}]".format(self.id,self.config.id,self.imgpath))
   
   def setImageMetadata(self):
      self.size = self.img.size
      self.width = self.img.shape[0]
      self.height = self.img.shape[1]
      if len(self.img.shape) > 2:
         self.channels = self.img.shape[2]
      else:
         self.channels = 1
      #print("channel********",self.channels)

   def getImage(self):
      return self.img

   def getImagePath(self):
      return self.imgpath

   def getImageSize(self):
      return self.size

   def getImageDim(self):
      return self.width, self.height

   def log(self,msg,methodName=""):
      sep = '|'
      self.logger.log( self.ekey + sep + self.key + '::' + methodName + sep + msg + sep)

   def showImage(self,imagekey="",img=None,imageid=None):
      mname = 'showImage' 
      
      if img is None:
        if imagekey != "":
          imgid = imagekey
          img = self.imgdict.get(imagekey,None)
        else:
          imgid = self.id
          img = self.img
      
       
      misc.imshow(imagekey,img)
      k = misc.waitKey(0)
      if k == 27:         # wait for ESC key to exit
         misc.destroyAllWindows()
      elif k == ord('s'): # wait for 's' key to save and exit
         #misc.imwrite('o.png',self.img)
         misc.destroyAllWindows()

   def printPixel(self,x0,y0,w=5,h=.10):
      mname = 'printPixel' 
      #w = h = 10
     
      if x0 < 0:
        x0 = 0 
      if y0 < 0:
        y0 = 0 

      print(self.img.shape)
      if len(self.img.shape) > 2:
        px = self.img[x0:(x0+w),y0:(y0+h),:]
      else:
        px = self.img[x0:(x0+w),y0:(y0+h)]
      #print('px - ' + px)
      #print(px.shape)
      px_patch = px
      self.draw_3d_plot(px_patch=px)
     
       
      if len(self.img.shape) > 2:
        np.savetxt('/tmp/px0.csv', px[:,:,0], fmt='%03d')
        np.savetxt('/tmp/px1.csv', px[:,:,1], fmt='%03d')
        np.savetxt('/tmp/px2.csv', px[:,:,2], fmt='%03d')
      else:
        np.savetxt('/tmp/px0.csv', px, fmt='%03d')
      px = px.astype('float32')
      #print(px)
      px /= 255.0
      #print(px)
      '''
      for i in range(3):
        #print(i,px[:,:,i].shape,np.mean(px[:,:,i]),px[:,:,i])
        print(i,px[:,:,i].shape,np.mean(px[:,:,i]))
        px[:,:,i] -= np.mean(px[:,:,i])
        px[:,:,i] /= np.std(px[:,:,i])
      '''
      px -= np.mean(px)
      px /= np.std(px)
      #self.draw_3d_plot(px_patch=px)
      if len(self.img.shape) > 2:
        np.savetxt('/tmp/pxm0.csv', px[:,:,0], fmt='%0.3f')
        np.savetxt('/tmp/pxm1.csv', px[:,:,1], fmt='%0.3f')
        np.savetxt('/tmp/pxm2.csv', px[:,:,2], fmt='%0.3f')
      else:
        np.savetxt('/tmp/pxm0.csv', px, fmt='%0.3f')
      px = np.around( px, decimals=3)
      #print(px)
      #print(self.img.item((x,y,1)))
     
      return px_patch

   def addRandNoise(self):
      mname = 'addRandNoise'
       
      img_noisy = util.random_noise(self.img,mode='gaussian',seed=None,clip=True,mean=0,var=0.01)
      self.img = self.img ** img_noisy
      #print(' noise - ' + str(img_noisy[5:6,93:105]))
    
   def printImageProp2(self):
      mname = 'printImageProp' 
      
      print('-------------------------------------------------------------------------------')
      print('      id         - {}'.format(self.id))
      print('      imgpath    - {}'.format(self.imgpath))
      print('      size       - {}'.format(self.size))
      print('      shape      - {} X {}'.format(str(self.width),str(self.height)))
      print('      rawshape   - {}'.format(self.img.shape))
      print('      pixel size - {}'.format(type(self.img)))
      imagekeys = self.imgdict.keys()
      for imagekey in imagekeys:
        print('        image[{}] - size[{}] shape[{}]'.format(imagekey,self.imgdict.get(imagekey).size,self.imgdict.get(imagekey).shape))
      #self.logger.log('----------------------------------------------------')

    
   def printImageProp(self):
      mname = 'printImageProp' 
      
      self.logger.log('-------------------------------------------------------------------------------')
      self.logger.log('      id         - {}'.format(self.id))
      self.logger.log('      imgpath    - {}'.format(self.imgpath))
      self.logger.log('      size       - {}'.format(self.size))
      self.logger.log('      shape      - {} X {}'.format(str(self.width),str(self.height)))
      self.logger.log('      rawshape   - {}'.format(self.img.shape))
      self.logger.log('      pixel size - {}'.format(type(self.img)))
      imagekeys = self.imgdict.keys()
      for imagekey in imagekeys:
        self.logger.log('        image[{}] - size[{}] shape[{}]'.format(imagekey,self.imgdict.get(imagekey).size,self.imgdict.get(imagekey).shape))
      #self.logger.log('----------------------------------------------------')

   def getImageFromSpecificChannel(self,channel=2,convertFlag = False):
      if self.channels == 3:
        if convertFlag:
          self.img = self.img[:,:,channel]
          #print(self.img.shape,"*******")
          self.setImageMetadata()
           
          return self.img
        else:
           
          return self.img[:,:,channel]
      else:
        #for existing single channel system, return img as is 
        return self.img

   def getGreyScaleImage2(self,convertFlag = False):
      red = .23
      blue = .07
      green = .7
      if self.channels == 3:
        if convertFlag:
          #self.img = np.average( self.img, axis=2)
          self.img = self.img[:,:,0] * red + self.img[:,:,1] * blue + self.img[:,:,2] * green
          self.setImageMetadata()
           
          return self.img
        else:
           
          #return np.average( self.img, axis=2)
          return self.img[:,:,0] * red + self.img[:,:,1] * blue + self.img[:,:,2] * green
      else:
        #for existing single channel system, return img as is 
        return self.img

   def getGreyScaleImage(self,convertFlag = False):
      if self.channels == 3:
        if convertFlag:
          self.img = np.average( self.img, axis=2)
          self.setImageMetadata()
           
          return self.img
        else:
           
          return np.average( self.img, axis=2)
      else:
        #for existing single channel system, return img as is 
        return self.img

   def padImage( self, n_img_w, n_img_h):
      i_w, i_h = self.getImageDim() 
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
       
      if self.channels == 3:
        croped_img_arr = np.zeros((n_img_w,n_img_h,self.channels),dtype='uint8') 
        croped_img_arr[ t_img_w_offset:(t_img_w_offset + calc_img_w), t_img_h_offset:(t_img_h_offset + calc_img_h), :] = self.getImage()[ s_img_w_offset:(s_img_w_offset + calc_img_w), s_img_h_offset:(s_img_h_offset + calc_img_h), :]
      else:
        croped_img_arr = np.zeros((n_img_w,n_img_h),dtype='uint8') 
        croped_img_arr[ t_img_w_offset:(t_img_w_offset + calc_img_w), t_img_h_offset:(t_img_h_offset + calc_img_h)] = self.getImage()[ s_img_w_offset:(s_img_w_offset + calc_img_w), s_img_h_offset:(s_img_h_offset + calc_img_h)]
       
      self.img = None
      self.img = croped_img_arr
      self.setImageMetadata()
       
      return True   
    
   def showImageAndHistogram(self,imagekeys=None,img_arr=None):
      #prepare keys for iterating all dictionary images.
      patch = False
      if imagekeys is None:
        imagekeys = self.imgdict.keys()
      if img_arr is None: #check if print patch of image
        img_arr = self.img
      else:
        patch = True
      #Create subplot to accomodate all images and historgram
      #uncomment to see plt graphs
      fig, axes = plt.subplots(nrows=len(imagekeys)+1,ncols=2) #init subplot 
      ax = axes.ravel()
       
      #put source or original image to display
      ax[0].imshow(img_arr)
      ax[0].set_title(self.id)
      #put histogram to display
      ax[1].hist(img_arr.ravel(),256,[0,256])
      ax[1].set_title('Histogram')
       
      #iterate dictionary
      for i,imagekey in enumerate(imagekeys):
        #put image to display
        print("i[%s] imagekey[%s]" % (i,imagekey))
        '''
        if patch:
          ax[i+2].imshow(img_arr)
        else:
          ax[i+2].imshow(self.imgdict.get(imagekey,None))
        ax[i+2].set_title(imagekey)
        #put histogram to display
        ax[i+3].hist(img_arr.ravel(),256,[0,256])
        ax[i+3].set_title(imagekey + ' Histogram')
        '''
        #put source or original image to display
        fig, axes = plt.subplots(nrows=1,ncols=2) #init subplot 
        ax = axes.ravel()
        temp_img_arr = self.imgdict.get(imagekey,None)
        ax[0].imshow(temp_img_arr)
        ax[0].set_title( imagekey + self.id)
        #put histogram to display
        ax[1].hist(temp_img_arr.ravel(),256,[0,256])
        ax[1].set_title(imagekey + 'Histogram')
        plt.tight_layout()
        plt.show()
      
      #plt.hist(img_arr.ravel(),256,[0,256]); 
      #uncomment to see plt graphs
      plt.tight_layout()
      plt.show()
   
   def getImageByKey(self,imagekey):
      return self.imgdict[imagekey]
    
   def saveImage(self,img=None,img_type_ext='.jpg',gen_new_filename=False,augmented_ext=None):
      ofile = ""
       
      if gen_new_filename:
        #print("##i_img_file_name[{}]".format(self.i_img_file_name))
        imgpath = ''
        img_file_name = self.i_img_file_name.split("/") 
        if len(img_file_name) > 1:
           imgpath = img_file_name[-1]
        else:
           imgpath = img_file_name[0]
        m1 = re.search("(^.*?)\.(\w+)$",imgpath)
        #ofile = self.config.odir + m1.group(1) + '_u' + img_type_ext
        if augmented_ext is None:
          ofile = self.config.odir + m1.group(1) + img_type_ext
        else:
          ofile = self.config.odir + m1.group(1) + augmented_ext + img_type_ext
      else:
        ofile = self.config.odir + 'cntr_' + self.id + img_type_ext
       
      if type(img).__name__ != self.config.typeNone: #img is passed so use it.
        #print("saveImage: Saving override Image.")
        misc.imwrite( ofile, img)
      else:
        #print("##image shape[{}]".format(self.getImage().shape))
        misc.imwrite( ofile, self.getImage())
       
      return ofile

   def writeDictImages(self):
      imagekeys = self.imgdict.keys()
      for imagekey in imagekeys:
         ofile = self.config.odir + self.id + '_' + imagekey + '.jpg'
         misc.imwrite( ofile ,self.imgdict.get(imagekey))

   def imageAveraging(self,neig,img):
      mname = 'imageAveraging' 
      
      self.logger.log('{} averaging using {}'.format(mname,str(neig)))
      self.logger.log('Averaing  - {} X {}'.format(neig,neig))
      x,y = img.shape
      #self.logger.log(' Image shape  - %d X %d' % (x,y))
      units =  math.floor(neig/2)
      imgSize = x*y
      x1 = x - 2*units
      y1 = y - 2*units
      self.logger.log(' processing index shape  - {} X {} '.format(x1,y1))
      t_img_ind = np.empty([x1,y1],dtype=int)
      ind = range(imgSize)
      for i in range(t_img_ind.shape[0]):
        s = int(y*(units+i)+units)
        t_img_ind[i,:] = ind[s:int(s+y1)]
      #self.logger.log(t_img_ind)

   def convoluteUseAverage(self,type,units,img,t_img_ind):
      mname = 'convoluteUseAverage' 
      
      self.logger.log( mname + ' averagin using ' + str(neig))
    
   def getBlurImage(self,imagekey=""):
     # downsample and use it for processing
     img = None
     
     if imagekey == "": #set imagekey if null
       imagekey = "blur"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else: 
       self.imgdict[imagekey] = misc.pyrDown(self.img)
       
       return self.imgdict[imagekey]
     
   def getGrayImage(self,imagekey=""):
     # apply grayscale
     img = None
     
     if imagekey == "": #set imagekey if null
       imagekey = "gray"
   
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else: 
       #img = self.img  #if blurring is not done then #no contours blow up 2.5 times.
       img = self.getBlurImage()
       self.imgdict[imagekey] = misc.cvtColor( img, misc.COLOR_BGR2GRAY)
       
       return self.imgdict[imagekey]
   
   '''     
    Apply errosion to make highlight thinner.
   '''     
   def getMorphErode(self,imagekey=""):
     # Add's one iteration of erosion to normal gray image.
     img = None
     #morph_kernel_mask = (3,3) #initialize morph kernel to 3 X 3
     morph_kernel_mask = np.ones((3,3) ,np.uint8) #initialize morph kernel to 3 X 3
     
     if imagekey == "": #set imagekey if null
       imagekey = "morpherode"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else:
       oimg = self.getGrayImage() #works on gray image. Actual it enhances write content
       eroded = misc.erode( oimg, morph_kernel_mask, iterations=1)
       ''' #commented as this code is useless. it produces emobossing.
       temp = misc.dilate( eroded, morph_kernel_mask)
       temp = misc.subtract( oimg, temp)
       skel = np.zeros( oimg.shape, np.uint8)
       skel = misc.bitwise_or( skel, temp)
       self.imgdict[imagekey] = skel
       '''
       self.imgdict[imagekey] = eroded
       
       return self.imgdict[imagekey]
     
   '''     
    Apply gradient change of thicken edges also called as dialetion.
   '''     
   def getMorphGradientImage(self,imagekey=""):
     # morphological gradient
     img = None
     morph_kernel_mask = (3,3) #initialize morph kernel to 3 X 3
     
     if imagekey == "": #set imagekey if null
       imagekey = "morphgradient"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else: 
       morph_kernel = misc.getStructuringElement(misc.MORPH_ELLIPSE, morph_kernel_mask )
       self.imgdict[imagekey] = misc.morphologyEx( self.getGrayImage(), misc.MORPH_GRADIENT, morph_kernel)
       
       #build image with key "emorphgradient" to represent gray image with one iteration using erode.
       self.imgdict['e' + imagekey] = misc.morphologyEx( self.getMorphErode(), misc.MORPH_GRADIENT, morph_kernel)
       
       return self.imgdict[imagekey]
     
   def getGBinaryImage(self,imagekey="",fromimagekey=""):
     # Gaussian binarize
     img = None
     
     if imagekey == "": #set imagkey if null
       imagekey = "gbinary"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else:
       if fromimagekey == "":
          #tmpimg = misc.GaussianBlur(self.getMorphGradientImage(),(3,3),0)
          tmpimg = self.getImageByKey(imagekey="emorphgradient")
          _, self.imgdict[imagekey] = misc.threshold( src=tmpimg, thresh=0, maxval=255, type=misc.THRESH_BINARY+misc.THRESH_OTSU)
          return self.imgdict[imagekey]
       else:
          #print("Pulling image from [{}]".format(fromimagekey))
          _, tmpimg = misc.threshold(src=self.getImageByKey(imagekey=fromimagekey), thresh=0, maxval=255, type=misc.THRESH_BINARY+misc.THRESH_OTSU)
          self.imgdict[imagekey] = tmpimg
          
          return tmpimg
       
   def getBinaryImage(self,imagekey="",fromimagekey=""):
     # binarize
     img = None
     
     if imagekey == "": #set imagkey if null
       imagekey = "binary"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else:
       if fromimagekey == "":
          _, self.imgdict[imagekey] = misc.threshold(src=self.getMorphGradientImage(), thresh=0, maxval=255, type=misc.THRESH_BINARY+misc.THRESH_OTSU)
          return self.imgdict[imagekey]
       else:
          _, tmpimg = misc.threshold(src=self.getImageByKey(imagekey=fromimagekey), thresh=0, maxval=255, type=misc.THRESH_BINARY+misc.THRESH_OTSU)
          return tmpimg
       
   def getHorizontalDialtedImageWithRect(self,imagekey=""):
     # connect horizontally oriented regions
     img = None
     morph_kernel_mask = (9,1)
     
     if imagekey == "": #set imagkey if null
       imagekey = "horizontaldialated"
    
     img = self.imgdict.get(imagekey,None)
     if img is not None: #check if blur image already in dict and if not create one
       return img
     else: 
       morph_kernel = misc.getStructuringElement(misc.MORPH_RECT, morph_kernel_mask)
       self.imgdict[imagekey] = misc.morphologyEx( self.getBinaryImage(), misc.MORPH_CLOSE, morph_kernel)
        
       return self.imgdict[imagekey]
    
   def add_text( self, text, x=0, y=0, image_scale=10):
      """
      Args:
          img (numpy array of shape (width, height, 3): input image
          text (str): text to add to image
          text_top (int): location of top text to add
          image_scale (float): image resize scale
  
      Summary:
          Add display text to a frame.
  
      Returns:
          Next available location of top text (allows for chaining this function)
      """
      u_img = misc.putText(
                    img=self.img,
                    text=text,
                    org=(x, y),
                    fontFace=misc.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.15 * image_scale,
                    color = (255, 0, 0),
                    thickness = 2)
      #self.saveImage(img=u_img,img_type_ext='.tif')
      self.saveImage(img_type_ext='.tif',gen_new_filename=True)
       
      return y + int(5 * image_scale)  

   def plot_3d( self, Z, w=None, h=None, title='surface'):
      
     if w is None:
       x = np.linspace( 0, self.width, self.width)
     else:
       x = np.linspace( 0, w, w)
      
     if h is None:
       y = np.linspace( 0, self.height, self.height)
     else:
       y = np.linspace( 0, h, h)
      
     X, Y = np.meshgrid(x,y) 
     print(" X - ",X.shape)
     print(" Y - ",Y.shape)
     print(" Z - ",Z.shape)
      
     fig = plt.figure()
     ax = plt.axes(projection="3d")
     #ax.plot_wireframe(X, Y, Z, color='green', rstride=10, cstride=10)
     ax.plot_wireframe(X, Y, Z, color='green')
     #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10,cmap='winter', edgecolor='none')
     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('z')
     ax.set_title(title)
      
     plt.show()
      
   def draw_3d_plot( self, px_patch=None):
      
     #Z = self.getGreyScaleImage2()
     if px_patch is not None:
        w = px_patch.shape[0] 
        h = px_patch.shape[1] 
        if self.channels == 1: 
          self.plot_3d(Z=px_patch,w=w,h=h,title='Greyscale=')
        else:
          for i in range(px_patch.shape[2]):
            self.plot_3d(Z=px_patch[:,:,i],w=w,h=h,title='channel=' + str(i))
     else:
       if self.channels == 1: 
         Z = self.img
         self.plot_3d(Z)
       else:
         for i in range(self.channels):
           Z = self.img[:,:,i]
           Z = Z.transpose(Z,(1,0))
           self.plot_3d(Z,title=self.id + ' channel=' + str(i))
         Z = self.getGreyScaleImage2()
         self.plot_3d(Z,title=self.id + ' Grayscale')
    
   def gen_augmented_images(self):
     aug_filenames = []
     img = self.img
     w = self.width
     h = self.height
     center = (w / 2, h / 2)
      
     angle90 = 90
     angle180 = 180
     angle270 = 270
      
     scale = 1.0
      
     # Perform the counter clockwise rotation holding at the center
     # 90 degrees
     M = misc.getRotationMatrix2D(center, angle90, scale)
     rotated90 = misc.warpAffine(img, M, (h, w))
     self.printImageProp2()
     o_file_name = self.saveImage( img=rotated90, gen_new_filename=True, augmented_ext='_r90')
     #aug_img = myImg( imageid=self.id, config=self.config, path=o_file_name) 
     #aug_img.printImageProp2()
     #self.showImage(img=rotated90,imageid='rotated90')
      
     # 180 degrees
     M = misc.getRotationMatrix2D(center, angle180, scale)
     rotated180 = misc.warpAffine(img, M, (w, h))
     #self.showImage(img=rotated180,imageid='rotated180')
      
     # 270 degrees
     M = misc.getRotationMatrix2D(center, angle270, scale)
     rotated270 = misc.warpAffine(img, M, (h, w))
     #self.showImage(img=rotated270,imageid='rotated270')
      
 

def main(argv=None):
   ''' 
   try:
      opts, args = getopt.getopt(argv,"ic:h",["i_img","i_cdir"])
   except getopt.GetoptError:
      print("test.py -i <input image file>")
      sys.exit(2)
   for opt, arg in opts:
      print(opt,arg)
      if opt == '-h':
         print("myImg.py -i <input image file>")
         print("myImg.py -c <home dir> -i <input image file>")
         sys.exit()
      elif opt in ("-i", "--i_img"):
         i_imgpath = arg
      elif opt in ("-c", "--i_cdir"):
         i_cdir = arg
   ''' 
    
   #i_imgpath = 'normal_fundus.jpg'
   i_imgpath = '/scratch/data/croped/10003_left.jpeg'
   i_cdir = '../'
   print("Input image file is ".format(i_imgpath))
   print("Input working directory is ".format(i_cdir))

   config = cutil.Config(configid="myConfId",cdir=i_cdir)
   img1 = myImg(imageid="xx",config=config,ekey='x123',path=i_imgpath)
   #img1.saveImage()
   img1.printImageProp()
   print("----------------------------------------------")
   '''
   img1.getHorizontalDialtedImageWithRect()
   img1.getGBinaryImage(fromimagekey="emorphgradient")
   img1.getMorphErode()
   img1.writeDictImages()
   '''
   #img2.showImageAndHistogram()


if __name__ == "__main__":
   #main(sys.argv[1:])
   i_imgpath = '/disk1/data1/data/train/15916_right.jpeg'
   #i_imgpath = '/data1/data/croped/15916_right.jpeg'
   #i_imgpath = '/tmp/15916_right.jpeg'
   #i_imgpath = '/Users/pankaj.petkar/dev/ret/dr_color/36181_left.jpeg'
   i_imgpath = '/tmp/patches/1100_left_mi_2212.jpeg'
   i_cdir = '/tmp/'
   print("Input image file is [{}]".format(i_imgpath))
   print("Input working directory is [{}]".format(i_cdir))

   config = cutil.Config(configid="myConfId",cdir=i_cdir)
   img1 = myImg(imageid="xx",config=config,ekey='x123',path=i_imgpath)
   #img1.saveImage()
   img1.printImageProp2()
   img1.showImageAndHistogram()
   '''
   img1.getHorizontalDialtedImageWithRect()
   img1.getGBinaryImage(fromimagekey="emorphgradient")
   #img1.getMorphErode()
   img1.showImageAndHistogram()
   '''
   #patch = img1.printPixel(x0=1772,y0=1547,w=70,h=70)
   #img1.showImageAndHistogram("my patch",patch)
   img1.printImageProp2()
   img1.draw_3d_plot()
   #img1.gen_augmented_images()
   
   ''' 
   img1.getImageFromSpecificChannel(channel=2,convertFlag=True) 
   img1.padImage(2500,2500)
   img1.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
   i_imgpath = '/tmp/img/15916_right.jpeg'
   print("Input image file is [{}]".format(i_imgpath))
   img2 = myImg(imageid="xx1",config=config,ekey='x123',path=i_imgpath,channels=1)
   img2.printImageProp2()
   ''' 
