import cv2
import numpy as np
import tensorflow as tf

# Function to extract patches using 'extract_image_patches'
def img_to_patches(raw_input, _patch_size=(128, 128), _stride=100):

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
def patches_to_img(update, _block_shape, _stride=100):
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

    
def reshape_img( n_img_w, n_img_h, imgbuf):
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
      croped_img_arr = np.zeros((n_img_w,n_img_h,1),dtype='uint8') 
      croped_img_arr[ t_img_w_offset:(t_img_w_offset + calc_img_w), t_img_h_offset:(t_img_h_offset + calc_img_h), :] = imgbuf[ s_img_w_offset:(s_img_w_offset + calc_img_w), s_img_h_offset:(s_img_h_offset + calc_img_h), :]
     
    return croped_img_arr
    
def get_img_patches(img,patch_size):
  # getinitial data
  p = patch_size #patch size
  h = tf.shape(img)[1] #image size
  c = tf.shape(img)[3] #image channels
   
  # Image to Patches Conversion
  pad = [[0,0],[0,0]]
  patches = tf.space_to_batch_nd(img,[p,p],pad)
  patches = tf.split(patches,p*p,0)
  patches = tf.stack(patches,3)
  patches = tf.reshape(patches,[(h/p)**2,p,p,c])
  #ueError: Cannot reshape a tensor with 12582912 elements to shape [0,128,128,2048] (0 elements) for 'Reshape' (op: 'Reshape') with input shapes: [1,16,16,16384,3], [4] and with input tensors computed as partial shapes: input[1] = [0,128,128,2048].
  
  return patches


def reconstruct_patches(patches,patch_size,image_size,channels):
  # getinitial data
  p = patch_size #patch size
  h = image_size #image size
  c = channels #image channels
  pad = [[0,0],[0,0]]
     
  # Do processing on patches
  # Using patches here to reconstruct
  patches_proc = tf.reshape(patches,[1,h/p,h/p,p*p,c])
  patches_proc = tf.split(patches_proc,p*p,3)
  patches_proc = tf.stack(patches_proc,axis=0)
  patches_proc = tf.reshape(patches_proc,[p*p,h/p,h/p,c])

  reconstructed_image = tf.batch_to_space_nd(patches_proc,[p, p],pad)
  
  return reconstructed_image

def test_img_ext_reconst(img_path,grayscale=False):
  patch_size = 128
  image_size = 2048
  
  print("Processing image [{}]".format(img_path))
  if grayscale: 
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
  else:
    image = cv2.imread(img_path,)
  print("image shape [{}]".format(image.shape))
  if len(image.shape) < 3:
    image = np.reshape( image, (image.shape[0], image.shape[1], 1))
  image = reshape_img( image_size, image_size, image)
  # Add batch dimension
  u_image = np.expand_dims(image, axis=0)
  print("croped image shape [{}]".format(image.shape))
  #cv2.imshow(img_path,image)
  #cv2.waitKey(0)
   
  o_img = tf.convert_to_tensor(u_image,dtype=tf.uint8)
  print("o_img tensor shape [{}]".format(tf.shape(o_img)))
  patches = get_img_patches( o_img, patch_size)
  reconstructed_image = reconstruct_patches(patches,patch_size,tf.shape(o_img)[1],tf.shape(o_img)[3])
   
  with tf.Session() as sess:
    P,R_n = sess.run([patches,reconstructed_image])
    I = image
    print(I.shape)
    print(P.shape)
    print(R_n.shape)
    err = np.sum((R_n-I)**2)
    print(err)
    R_n = np.squeeze(R_n, axis=0)
    if len(image.shape) < 3:
      R_n = np.reshape( R_n, (image.shape[0], image.shape[1]))
    print(R_n.shape)
    #cv2.imshow('Reconstructed Image',R_n)
    #cv2.waitKey(0)

if __name__ == "__main__":
  test_img_ext_reconst(img_path='/tmp/ex/images/IDRiD_50.jpg')
  test_img_ext_reconst(img_path='/tmp/ex/gt/IDRiD_50_HE.jpg',grayscale=True)
