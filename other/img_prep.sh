#!/bin/bash

ddir="/data1/data"
wdir="/tmp"
cdir=`pwd`
erfile="dr_err.txt"

error_file="$wdir/$erfile"

batch_size=2500
no_files=35126
no_batch=$(($no_files/$batch_size))
last_batch_start=$(($no_batch*$batch_size))
last_batch_size=$(($no_files%$batch_size))

batch_ind=0

line_break() {
  echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
}

line_break
echo "  Images to process [$no_files]"
echo "  Batch size [$batch_size]"
echo "  Batches to run [$no_batch]"
echo "  Start of last batch [$last_batch_start]"
echo "  Last batch size [$last_batch_size]"
line_break

execute_cmd() {
  if [ $# -eq 0 ]
  then
    echo "execute_cmd called with cmd text null"
    exit -99
  fi
  cmd=$*
  cmd="nohup $cmd 1> $wdir/pp_err$batch_ind.txt 2>&1 &"
  echo "Executing command[$cmd]"
  error=$(eval $cmd)
  ret=$?
  if [ $ret -ne 0 ]; then
      echo "Fail execution of $cmd failed with ret[$ret $error..."
      exit -98
  fi
}

start_index=1
#for i in {1..$no_batch}
#no_batch=2
for ((i=0; i<$no_batch; i++))
do
  from=$start_index
  to=$(($start_index+$batch_size))
  #echo "In loop $i processing $from to $to"
  cmd="python $SRC/utils/data_util.py --from=$from --batch_size=$batch_size"
  batch_ind=$(($batch_ind+1))
  execute_cmd $cmd
   
  start_index=$to
done

#echo "last batch $start_index to $no_files"
line_break

cmd="python $SRC/utils/data_util.py --from=$start_index --batch_size=$last_batch_size"
execute_cmd $cmd
line_break

