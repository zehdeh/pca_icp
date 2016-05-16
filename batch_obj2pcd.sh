#!/bin/bash
FILES="${1}/*.obj"
TARGETFOLDER="${2}"

for f in $FILES 
do
	filename="$(basename $f)"
	filename_noext="${filename%.*}"
	targetname="$TARGETFOLDER/$filename_noext.pcd"
	/usr/local/bin/pcl_obj2pcd $f $targetname -copy_normals1
done
