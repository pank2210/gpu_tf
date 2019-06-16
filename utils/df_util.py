
import pandas as pd
import os
import sys

sys.path.append('../') 
  
def print_groups(i_file,g_count_key='prob',g_keys=['label','pred'],g_sort_keys=['label','pred']):
  fname = 'print_groups'
   
  df = pd.read_csv( i_file)
  cols = df.columns.tolist()
  print("Processing file[%s] key[%s] g_keys[%s] sort[%s] columns[%s]" % (i_file,g_count_key,g_keys,g_sort_keys,cols))
  if cols[1] != 'level' and cols[1] != 'label':
    df.columns = ['id','level'] 
  #df.intent = df.intent.fillna('NA')
  g_df = df[ \
      #(df.status_code != 200) & \
      (df[g_count_key] >= 0) \
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
  key = sys.argv[2]
  print("Processing [%s]" % (i_file))
  fp = os.path.isfile(i_file)
  if fp:
    print_groups( i_file, g_count_key=key, g_keys=[key], g_sort_keys=[key])
  
