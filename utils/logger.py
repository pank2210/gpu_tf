
import sys
import logging

class myLogger:
  def __init__(self,logFileName,level=0):
    #initiatize
    #print("myLogger.init")
    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    logging.basicConfig(format=FORMAT,filename=logFileName,filemode='a',level=logging.INFO)
    self.d = {'clientip': '127.0.0.1', 'user': 'pp123'}
    self.logger = logging.getLogger(logFileName)
    #self.logger = logging.getLogger('tcpserver')

  def log(self,msg,level=0):
    #log 
    self.logger.info(msg,extra=self.d)

if __name__ == "__main__":
  myL = myLogger(logFileName="/tmp/myApp.log")
  myL.log("My first log")

