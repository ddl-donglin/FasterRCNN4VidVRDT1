# Faster-RCNN Pytorch

## Prepare
```bash
source activate pytorch
pip install -r requirements.txt
# Compile the cuda dependencies
cd lib
bash make.sh    # mayb u need 2 modify 'CUDA_ARCH' 2 suit u gpu version 

```
## Train
### VOC 2007
Download pretrained models:

[vgg16](https://drive.google.com/open?id=1Jg2G8LM3NMSZJovioVIynqDKrEBhVGsR)

[resnet101](https://drive.google.com/open?id=1i-o5YeRjiAeQPAR7EbHrnKQjvlu4nZGD)

```bash
bash prepare_voc2007.sh
bash gpu_res101.sh
# or
bash gpu_vgg16.sh

```

### Vidor_10k
Download [Grand Challenge dataset](http://lms.comp.nus.edu.sg/research/dataset.html)

```bash

```

### Modify
Check u own proj structure with [tree.txt](tree.txt) 2 modify
