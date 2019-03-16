# Faster-RCNN Pytorch

## Authorship
This project is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

Modified some parts to be the 1st step of Video VRD project.

## Benchmarking
[Benchmarking](https://github.com/jwyang/faster-rcnn.pytorch)

## Prepare
```bash
source activate pytorch
pip install -r requirements.txt
# Compile the cuda dependencies
cd lib
bash make.sh    # mayb u need 2 modify 'CUDA_ARCH' 2 suit u gpu version 

```
## Train
#### VOC 2007
Download pretrained models:

[vgg16](https://drive.google.com/open?id=1Jg2G8LM3NMSZJovioVIynqDKrEBhVGsR)

[resnet101](https://drive.google.com/open?id=1i-o5YeRjiAeQPAR7EbHrnKQjvlu4nZGD)

```bash
bash prepare_voc2007.sh
bash gpu_res101.sh
# or
bash gpu_vgg16.sh

```

#### Vidor_10k
Download [Grand Challenge dataset](http://lms.comp.nus.edu.sg/research/dataset.html)

```bash

```

### Modify
Check u own proj structure with [tree.txt](tree.txt) 2 modify


## Test
#### VOC 2007
Evaluate the detection performance of a pre-trained vgg16 model on pascal_voc test set

```bash
bash gpu_test.sh
```

## Demo
#### VOC 2007
If you want to run detection on your own images with a pre-trained model, download the pretrained model listed in above tables or train your own models at first, then add images to folder $ROOT/images, and then run
```bash
bash gpu_demo.sh
```

## Webcam
You can use a webcam in a real-time demo by running
```bash
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy \
               --webcam $WEBCAM_ID
```
The demo is stopped by clicking the image window and then pressing the 'q' key.