提供一些参考资料
* YOLOv3-spp better than YOLOv3 - mAP = 60.6%, FPS = 20: https://pjreddie.com/darknet/yolo/
* Yolo v3 source chart for the RetinaNet on MS COCO got from Table 1 (e): https://arxiv.org/pdf/1708.02002.pdf
* Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
* Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg

## "You Only Look Once: Unified, Real-Time Object Detection (versions 2 & 3)"

A Yolo cross-platform Windows and Linux version (for object detection). Contributors: https://github.com/AlexeyAB/darknet/graphs/contributors

This repository is forked from Linux-version: https://github.com/pjreddie/darknet

More details: http://pjreddie.com/darknet/yolo/

This repository supports:

* both Windows and Linux
* both OpenCV 2.x.x and OpenCV <= 4.0
* both cuDNN >= v7
* CUDA >= 8.0
* also create SO-library on Linux and DLL-library on Windows


无人机视角的数据集下载地址：链接：https://pan.baidu.com/s/1r_4OTJJlGxo2fcBw4mr9lA 
提取码：vcwm ，其中，数据集分为四种场景，通过下载每一类场景数据集按比例整合成训练数据集和测试集等。

通过优化方法得到的最佳表现权重为anchor为9，通过样本分类、随机多尺度变化、不同的anchor机制以及难样本增强4种方法得到的权重下载地址：
链接：https://pan.baidu.com/s/1BDC_7MzjgRmaCyZ0JXo_MQ 
提取码：745h

通过改进YOLOv3模型结构和残差单元的改进型YOLOv3训练得到的最优权重下载地址：
链接：https://pan.baidu.com/s/1Z-pG8DAgVr02o_fClM1P6Q 
提取码：t5vn

YOLOV3使用方法可以参考官网给出的环境配置和使用方法，CFG文件中yolov3-darknet77-resunit2.cfg为改进后的CFG文件，做不同的实验室请选择不同的cfg文件进行。


