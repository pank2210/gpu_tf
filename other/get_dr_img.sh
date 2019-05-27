#!/bin/bash -x

ddir="/data1/data"
wdir="/tmp"
cdir=`pwd`
drdir="dr_img"
cp_cmd_file='cp_dr_files.sh'
erfile="dr_err.txt"

label_file="trainLabels.csv"
s_dir="$ddir/croped/"
t_dir="$wdir/$drdir/"
error_file="$t_dir/$erfile"
zip_file="$t_dir/dr_files.zip"

label_fl="$ddir/$label_file"

if [ -f $label_fl ]; then
  echo "Processing label file[$label_fl] for moving DR images."
else
  echo "$label_fl Main training label file absent, exiting the code."
  exit -1
fi

if [ -d $s_dir ]; then
  echo "Using $s_dir as source directory."
else
  echo "$s_dir source directory absent, so exiting as error."
  exit -1
fi

if [ -d $t_dir ]; then
  echo "Using $t_dir as target directory."
else
  echo "$t_dir target directory absent. Creating it..."
  mkdir $t_dir
  if [ $? -ne 0 ]; then
      echo "Fail creating $t_dir"
      exit -3
  fi
fi

#echo "$s_dir" | sed -e 's/\//\\\//g' | (read s_dir_u)
s_dir_u=`echo "$s_dir" | sed -E 's:\/:\\\/:g'`
t_dir_u=`echo "$t_dir" | sed -E 's:\/:\\\/:g'`
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo "Executing sed command[$sed_cmd]"
grep "[123]$" $label_fl | cut -d ',' -f 1 | sed -E 's/^(.*)$/cp '$s_dir_u'\1.jpeg '$t_dir_u'/g' > $t_dir/$cp_cmd_file 2> $error_file
if [ $? -ne 0 ]; then
    echo "Fail execution of sed command"
    exit -4
fi
echo "Executing all dr copy command script"
cmd="chmod 755 $t_dir/$cp_cmd_file"
error=$(eval "$cmd" 2>> "$error_file")
cmd=". $t_dir/$cp_cmd_file" 
error=$(eval "$cmd" 2>> "$error_file")
if [ $? -ne 0 ]; then
    echo "Fail execution of $cmd failed with $error..."
    exit -5
fi
echo "Execuitng zip command"
cd $wdir
zip -r $zip_file ./$drdir/*.jpeg 2>> $error_file 2>> $error_file
cd $cdir

if [ -f $zip_file ]; then
  echo "Target zip file [$zip_file] created."
else
  echo "Fail/Error in zip file [$zip_file] creation."
fi
