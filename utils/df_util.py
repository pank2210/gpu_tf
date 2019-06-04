
import pandas as pd
import os
import sys

sys.path.append('../') 
  
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

if __name__ == "__main__":
  '''
  if sys.argc <= 1:
    print("input argument file name absent")
    exit(-1)
  '''
  i_file = sys.argv[1]
  fp = os.path.isfile(i_file)
  if fp:
    print_groups( i_file, g_count_key='level', g_keys=['level'], g_sort_keys=['level'])
  
