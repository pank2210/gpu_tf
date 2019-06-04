import sys, getopt

'''
  Function to processing custom arguments.
'''
def get_preprocessing_index(argv):
   _from = 0
   _batch_size = 100
   try:
      opts, args = getopt.getopt(argv,"hf:b",["from=","batch_size="])
   except getopt.GetoptError:
      print('<utility> --from <start index> --to <batch size>')
      print('return default values i.e. from=0 and batch_size=100')
       
      return _from, _batch_size
      #sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('<utility> --from <start index> --batch_size <batch size>')
         sys.exit()
      elif opt in ("-f", "--from"):
         _from = int(arg)
      elif opt in ("-b", "--batch_size"):
         _batch_size = int(arg)
    
   print(" Entered options...")
   print("	_from	: %d" % (_from))
   print("	_batch_size	: %d" % (_batch_size))
    
   return _from, _batch_size

if __name__ == "__main__":
  get_preprocessing_index(sys.argv[1:])
