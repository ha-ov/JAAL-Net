# JAAL-Net: A Joint Attention and Adversarial Learning Network for Skin Lesion Segmentation


## Setup

1) You need download the datasets(ISBI2016，ISBI2017 and ISIC2018)
    - You can download the datasets from https://github.com/GalAvineri/ISIC-Archive-Downloader 
    - You can also download the datasets from https://challenge.isic-archive.com/data/#2016
      
   At the end, the directory of the data should be like this:
   
    ```
    Dataset/
    ├── Train/  (containing the train images)
    ├── Test/  (containing the test images)
    └── Validate/  (containing the validate images)
    ```

2) Train the model: `python3 train.py` 
    - default set -epoch 300 -lr 0.0002 -batch size 4
    - A model is generated for every 10 training rounds, and the model is stored in result/checkpoint/

3) Test the model: `python test.py --model 'result/checkpoint/netG_model_epoch_300.pth'`
    - The test evaluation indexes are Dice, Iou, Sensitivity and Accuracy, and the predicted images are stored in result/cGAN/


## Results

Original image

<img src="https://github.com/ha-ov/JAAL-Net/blob/main/example/c.jpg" width="200" height="200">

Classify and Segment image

<img src="https://github.com/ha-ov/JAAL-Net/blob/main/example/c-1.jpg" width="200" height="200">
