#!/usr/bin/env bash

if [[ ! -d data  ]];then
  mkdir data && cd $_
  mkdir pretrained_model output VOCdevkit2007
else
  echo data exist
  cd data
  mkdir pretrained_model output VOCdevkit2007
fi

# prepare the voc 2007 data

cd VOCdevkit2007

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar

mkdir VOC2007
mv VOCdevkit/VOC2007/* VOC2007/*

rm -rf VOCdevkit/


# prepare the pretrained model
# download from readme links
