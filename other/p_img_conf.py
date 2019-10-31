
import sys
import os
import json

import pandas as pd
import numpy as np

sys.path.append('../')

from utils import cmdopt_util as cmd_util

class ImgConf:
  
  def __init__(self,ifname,ofname):
    self.conf_dir = '../config/'
    self.ofname = ofname
    
    #var's required for output processing
    self.col_names = [ 'filename',
                       'shape_attributes.name',
                       'shape_attributes.x',
                       'shape_attributes.y',
                       'shape_attributes.width',
                       'shape_attributes.height',
                       'region_attributes.Name']
    self.o_sep = ','
    self.df = pd.DataFrame(columns=self.col_names)
    self.df_dict = {}
    for col in self.col_names:
      self.df_dict[col] = []
     
    self.ifd = open( self.conf_dir + ifname, 'r')
    #self.ofd = open( self.conf_dir + ofname, 'w')
    self.jdoc = json.load(self.ifd)
    
    self.process_data()
   
  def parse_dict(self,_dict):
    '''
      "1100_left.jpeg1470584": {
        "filename": "1100_left.jpeg",
        "size": 1470584,
        "regions": [
          {
            "shape_attributes": {
              "name": "rect",
              "x": 1496,
              "y": 1354,
              "width": 94,
              "height": 73
            },
            "region_attributes": {
              "Name": "MA"
            }
          },
    '''
    c_names = self.col_names
    
    filename = _dict[c_names[0]]
    for region in _dict['regions']:
      r_dict = {}
      r_dict[c_names[0]] = _dict[c_names[0]]
      
      try: 
        shape_type = region[c_names[1].split('.')[0]][c_names[1].split('.')[1]]
        if shape_type == 'polygon':
          all_x = np.array(region[c_names[1].split('.')[0]]["all_points_x"])
          all_y = np.array(region[c_names[1].split('.')[0]]["all_points_y"])
          r_dict[c_names[1]] = region[c_names[1].split('.')[0]][c_names[1].split('.')[1]]
          r_dict[c_names[2]] = np.amin(all_x)
          r_dict[c_names[3]] = np.amin(all_y)
          r_dict[c_names[4]] = np.amax(all_x) - np.amin(all_x)
          r_dict[c_names[5]] = np.amax(all_y) - np.amin(all_y)
          r_dict[c_names[-1]] = region[c_names[-1].split('.')[0]][c_names[-1].split('.')[1]]
        else:
          for c_name in c_names[1:]:
            r_dict[c_name] = region[c_name.split('.')[0]][c_name.split('.')[1]]
      except:
        print("Error reading json file. at [%s]" % (filename))
        print(sys.exc_info())
        sys.exit(-2)
      else:
        for col in c_names:
          arr = self.df_dict[col]
          arr.append(r_dict[col])
          self.df_dict[col] = arr
     
    #print("+++++++",r_dict)
   
  def gen_output(self):
    for col in self.col_names:
      self.df[col] = self.df_dict[col]
    self.df.to_csv(self.conf_dir + self.ofname, sep=self.o_sep, index=False)
     
    fnames = self.df.filename.unique()
    with open(self.conf_dir + 'labeled_images.txt','w') as f:
      f.writelines("%s\n" % item for item in fnames)
   
  def process_data(self): 
    for i,item in enumerate(self.jdoc):
      #if i > 3:
      #  break
      #print(i,type(item),item,type(self.jdoc[item]))
      self.parse_dict(self.jdoc[item])
    self.gen_output()
    
if __name__ == "__main__":
  ifname, ofname = cmd_util.get_imgconf_params(sys.argv[1:])
  ic = ImgConf(ifname,ofname)
