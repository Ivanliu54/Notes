# Tasks of Machine Vision
<!-- TOC -->

- [Tasks of Machine Vision](#tasks-of-machine-vision)
    - [一. 任务概述](#一-任务概述)
        - [1. Image Classification (分类)](#1-image-classification-分类)
        - [2. Object localization (目标定位)](#2-object-localization-目标定位)
        - [3. Objdect detection （目标检测）](#3-objdect-detection-目标检测)
        - [4. Semantic segmentation（语义分割）](#4-semantic-segmentation语义分割)
        - [5. Instance segmentation (实例分割)](#5-instance-segmentation-实例分割)
        - [6. Panoramica Sementation (全景分割)](#6-panoramica-sementation-全景分割)
    - [二. 数据库](#二-数据库)
        - [1. Classification](#1-classification)
        - [2. Object detection](#2-object-detection)
    - [三. 常用网络](#三-常用网络)
        - [1. Classification](#1-classification-1)
            - [图像分类经典网络结构](#图像分类经典网络结构)
                - [LeNet-5](#lenet-5)
                - [AlexNet](#alexnet)
                - [VGG-16/VGG-19](#vgg-16vgg-19)
                - [GoogLeNet](#googlenet)
                - [Inception v3/v4](#inception-v3v4)
                - [ResNet](#resnet)
                - [preResNet](#preresnet)
                - [ResNeXt](#resnext)
                - [随机深度](#随机深度)
                - [DenseNet](#densenet)
                - [下表对比了上述几种网络结构](#下表对比了上述几种网络结构)
        - [2. Object detection](#2-object-detection-1)
            - [基本思路(滑窗)](#基本思路滑窗)
            - [基于候选区域](#基于候选区域)
                - [R-CNN](#r-cnn)
                - [Fast R-CNN](#fast-r-cnn)
                - [Faster R-CNN](#faster-r-cnn)
                - [R-FCN](#r-fcn)
                - [小结](#小结)
            - [基于直接回归](#基于直接回归)
                - [YOLO](#yolo)
                - [SSD](#ssd)
                - [FPN](#fpn)
                - [RetinaNet](#retinanet)
            - [目标检测常用技巧](#目标检测常用技巧)
                - [非最大抑制(non-max suppression, NMS)](#非最大抑制non-max-suppression-nms)
                - [在线困难样例挖掘(online hard example mining, OHEM)](#在线困难样例挖掘online-hard-example-mining-ohem)
                - [在对数空间回归](#在对数空间回归)
    - [四. 参考文献](#四-参考文献)

<!-- /TOC -->
## 一. 任务概述

### 1. Image Classification (分类)

关注图像整体（人脸识别）

### 2. Object localization (目标定位)

在图像分类的基础上，我们还想知道图像中的目标具体在图像的什么位置，通常是以包围盒的(bounding box)形式。



基本思路 

多任务学习，网络带有两个输出分支。一个分支用于做图像分类，即全连接+softmax判断目标类别，和单纯图像分类区别在于这里还另外需要一个“背景”类。另一个分支用于判断目标位置，即完成回归任务输出四个数字标记包围盒位置(例如中心点横纵坐标和包围盒长宽)，该分支输出结果只有在分类分支判断不为“背景”时才使用。



人体位姿定位/人脸定位 

目标定位的思路也可以用于人体位姿定位或人脸定位。这两者都需要我们对一系列的人体关节或人脸关键点进行回归。 



弱监督定位 

由于目标定位是相对比较简单的任务，近期的研究热点是在只有标记信息的条件下进行目标定位。其基本思路是从卷积结果中找到一些较高响应的显著性区域，认为这个区域对应图像中的目标。


### 3. Objdect detection （目标检测）

图片里有什么？分别在哪？在目标定位中，通常只有一个或固定数目的目标，而目标检测更一般化，其图像中出现的目标种类和数目都不定。因此，目标检测是比目标定位更具挑战性的任务。

主要方法： Faster R-cnn / YOLO

### 4. Semantic segmentation（语义分割）

区分每一个像素属于哪一类，但是不区分同一类下的不同个体。
语义分割是目标检测更进阶的任务，目标检测只需要框出每个目标的包围盒，语义分割需要进一步判断图像中哪些像素属于哪个目标。



(1) 语义分割常用数据集



PASCAL VOC 2012 1.5k训练图像，1.5k验证图像，20个类别(包含背景)。

MS COCO COCO比VOC更困难。有83k训练图像，41k验证图像，80k测试图像，80个类别。



(2) 语义分割基本思路



基本思路 

逐像素进行图像分类。我们将整张图像输入网络，使输出的空间大小和输入一致，通道数等于类别数，分别代表了各空间位置属于各类别的概率，即可以逐像素地进行分类。



全卷积网络+反卷积网络 

为使得输出具有三维结构，全卷积网络中没有全连接层，只有卷积层和汇合层。但是随着卷积和汇合的进行，图像通道数越来越大，而空间大小越来越小。要想使输出和输入有相同的空间大小，全卷积网络需要使用反卷积和反汇合来增大空间大小。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0WkI8fTtuWqjgt5qae6akFUg7OYfOTq9KMrDiakSr9eyyddlYia63Kr6w/640?wx_fmt=jpeg)


反卷积(deconvolution)/转置卷积(transpose convolution) 

标准卷积的滤波器在输入图像中进行滑动，每次和输入图像局部区域点乘得到一个输出，而反卷积的滤波器在输出图像中进行滑动，每个由一个输入神经元乘以滤波器得到一个输出局部区域。反卷积的前向过程和卷积的反向过程完成的是相同的数学运算。和标准卷积的滤波器一样，反卷积的滤波器也是从数据中学到的。



反最大汇合(max-unpooling) 

通常全卷积网络是对称的结构，在最大汇合时需要记下最大值所处局部区域位置，在对应反最大汇合时将对应位置输出置为输入，其余位置补零。反最大汇合可以弥补最大汇合时丢失的空间信息。反最大汇合的前向过程和最大汇合的反向过程完成的是相同的数学运算。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0C0txUaWqssSaN3gsUPyStKQyXD8wibfRGIEtJeZaVF4MZkn2G7H4RlQ/640?wx_fmt=jpeg)




(3) 语义分割常用技巧



扩张卷积(dilated convolution) 

经常用于分割任务以增大有效感受野的一个技巧。标准卷积操作中每个输出神经元对应的输入局部区域是连续的，而扩张卷积对应的输入局部区域在空间位置上不连续。扩张卷积保持卷积参数量不变，但有更大的有效感受野。

 ![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0BMd8jLYsa3CppPyt40V8BXvEibrNicnq3IUGzQXdJIKZ1V9BSQkM3ic3Q/640?wx_fmt=jpeg)




条件随机场(conditional random field, CRF) 

条件随机场是一种概率图模型，常被用于微修全卷积网络的输出结果，使细节信息更好。其动机是距离相近的像素、或像素值相近的像素更可能属于相同的类别。此外，有研究工作用循环神经网络(recurrent neural networks)近似条件随机场。条件随机场的另一弊端是会考虑两两像素之间的关系，这使其运行效率不高。



利用低层信息 

综合利用低层结果可以弥补随着网络加深丢失的细节和边缘信息。

### 5. Instance segmentation (实例分割)

目标检测和语义分割的集合，相对于目标检测（画框），实例分割精度到物体边缘；相对于语义分割，实例分割需要比哦啊住处同一类物体的不同实例。

语义分割不区分属于相同类别的不同实例。例如，当图像中有多只猫时，语义分割会将两只猫整体的所有像素预测为“猫”这个类别。与此不同的是，实例分割需要区分出哪些像素属于第一只猫、哪些像素属于第二只猫。

基本思路 

目标检测+语义分割。先用目标检测方法将图像中的不同实例框出，再用语义分割方法在不同包围盒内进行逐像素标记。

常用方法：Mask R-CNN

Mask R-CNN 通过向 Faster R-CNN 添加一个分支来进行像素级分割，该分支输出一个二进制掩码，该掩码表示给定像素是否为目标对象的一部分：该分支是基于卷积神经网络特征映射的全卷积网络。将给定的卷积神经网络特征映射作为输入，输出为一个矩阵，其中像素属于该对象的所有位置用 1 表示，其他位置则用 0 表示，这就是二进制掩码。

一旦生成这些掩码， Mask R-CNN 将 RoIAlign 与来自 Faster R-CNN 的分类和边界框相结合，以便进行精确的分割：


Mask R-CNN 

用FPN进行目标检测，并通过添加额外分支进行语义分割(额外分割分支和原检测分支不共享参数)，即Master R-CNN有三个输出分支(分类、坐标回归、和分割)。此外，Mask R-CNN的其他改进有：(1). 改进了RoI汇合，通过双线性差值使候选区域和卷积特征的对齐不因量化而损失信息。(2). 在分割时，Mask R-CNN将判断类别和输出模板(mask)这两个任务解耦合，用sigmoid配合对率(logistic)损失函数对每个类别的模板单独处理，取得了比经典分割方法用softmax让所有类别一起竞争更好的效果。

### 6. Panoramica Sementation (全景分割)

全景分割是语义分割和实例分割的结合。跟实例分割不同的是：实例分割只对图像中的object进行检测，并对检测到的object进行分割，而全景分割是对图中的所有物体包括背景都要进行检测和分割

![Alt pic](http://5b0988e595225.cdn.sohucs.com/images/20180124/f72034dd098144658a4a718ac8d0b451.jpeg "图片")


## 二. 数据库

### 1. Classification 

以下是几种常用分类数据集，难度依次递增。

MNIST 60k训练图像、10k测试图像、10个类别、图像大小1×28×28、内容是0-9手写数字。
CIFAR-10 50k训练图像、10k测试图像、10个类别、图像大小3×32×32。
CIFAR-100 50k训练图像、10k测试图像、100个类别、图像大小3×32×32。
ImageNet 1.2M训练图像、50k验证图像、1k个类别。2017年及之前，每年会举行基于ImageNet数据集的ILSVRC竞赛，这相当于计算机视觉界奥林匹克。

### 2. Object detection


PASCAL VOC 包含20个类别。通常是用VOC07和VOC12的trainval并集作为训练，用VOC07的测试集作为测试。
MS COCO COCO比VOC更困难。COCO包含80k训练图像、40k验证图像、和20k没有公开标记的测试图像(test-dev)，80个类别，平均每张图7.2个目标。通常是用80k训练和35k验证图像的并集作为训练，其余5k图像作为验证，20k测试图像用于线上测试。
mAP (mean average precision) 目标检测中的常用评价指标，计算方法如下。当预测的包围盒和真实包围盒的交并比大于某一阈值(通常为0.5)，则认为该预测正确。对每个类别，我们画出它的查准率-查全率(precision-recall)曲线，平均准确率是曲线下的面积。之后再对所有类别的平均准确率求平均，即可得到mAP，其取值为[0, 100%]。
交并比(intersection over union, IoU) 算法预测的包围盒和真实包围盒交集的面积除以这两个包围盒并集的面积，取值为[0, 1]。交并比度量了算法预测的包围盒和真实包围盒的接近程度，交并比越大，两个包围盒的重叠程度越高


## 三. 常用网络


***基本架构*** 我们用conv代表卷积层、bn代表批量归一层、pool代表汇合层。最常见的网络结构顺序是conv -> bn -> relu -> pool，其中卷积层用于提取特征、汇合层用于减少空间大小。随着网络深度的进行，图像的空间大小将越来越小，而通道数会越来越大。

***针对你的任务，如何设计网络？*** 当面对你的实际任务时，如果你的目标是解决该任务而不是发明新算法，那么不要试图自己设计全新的网络结构，也不要试图从零复现现有的网络结构。找已经公开的实现和预训练模型进行微调。去掉最后一个全连接层和对应softmax，加上对应你任务的全连接层和softmax，再固定住前面的层，只训练你加的部分。如果你的训练数据比较多，那么可以多微调几层，甚至微调所有层。


### 1. Classification

#### 图像分类经典网络结构



基本架构 我们用conv代表卷积层、bn代表批量归一层、pool代表汇合层。最常见的网络结构顺序是conv -> bn -> relu -> pool，其中卷积层用于提取特征、汇合层用于减少空间大小。随着网络深度的进行，图像的空间大小将越来越小，而通道数会越来越大。



针对你的任务，如何设计网络？ 当面对你的实际任务时，如果你的目标是解决该任务而不是发明新算法，那么不要试图自己设计全新的网络结构，也不要试图从零复现现有的网络结构。找已经公开的实现和预训练模型进行微调。去掉最后一个全连接层和对应softmax，加上对应你任务的全连接层和softmax，再固定住前面的层，只训练你加的部分。如果你的训练数据比较多，那么可以多微调几层，甚至微调所有层。


##### LeNet-5 
60k参数。网络基本架构为：conv1 (6) -> pool1 -> conv2 (16) -> pool2 -> fc3 (120) -> fc4 (84) -> fc5 (10) -> softmax。括号中的数字代表通道数，网络名称中有5表示它有5层conv/fc层。当时，LeNet-5被成功用于ATM以对支票中的手写数字进行识别。LeNet取名源自其作者姓LeCun。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0KRAPPabeMyhgs5EXovOxvBAHv4QPbWAAXl3OKia2Riaice7XaezoJhvwA/640?wx_fmt=jpeg)

##### AlexNet 

60M参数，ILSVRC 2012的冠军网络。网络基本架构为：conv1 (96) -> pool1 -> conv2 (256) -> pool2 -> conv3 (384) -> conv4 (384) -> conv5 (256) -> pool5 -> fc6 (4096) -> fc7 (4096) -> fc8 (1000) -> softmax。AlexNet有着和LeNet-5相似网络结构，但更深、有更多参数。conv1使用11×11的滤波器、步长为4使空间大小迅速减小(227×227 -> 55×55)。AlexNet的关键点是：(1). 使用了ReLU激活函数，使之有更好的梯度特性、训练更快。(2). 使用了随机失活(dropout)。(3). 大量使用数据扩充技术。AlexNet的意义在于它以高出第二名10%的性能取得了当年ILSVRC竞赛的冠军，这使人们意识到卷机神经网络的优势。此外，AlexNet也使人们意识到可以利用GPU加速卷积神经网络训练。AlexNet取名源自其作者名Alex。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0Kdsw1iaNukJDk7pUDxicHTtMfYibxQJe3p0hwicDgyl8y0qyhDgZmre8kg/640?wx_fmt=jpeg)


##### VGG-16/VGG-19 

138M参数，ILSVRC 2014的亚军网络。VGG-16的基本架构为：conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3 -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> pool5 -> fc6 (4096) -> fc7 (4096) -> fc8 (1000) -> softmax。 ^3代表重复3次。VGG网络的关键点是：(1). 结构简单，只有3×3卷积和2×2汇合两种配置，并且重复堆叠相同的模块组合。卷积层不改变空间大小，每经过一次汇合层，空间大小减半。(2). 参数量大，而且大部分的参数集中在全连接层中。网络名称中有16表示它有16层conv/fc层。(3). 合适的网络初始化和使用批量归一(batch normalization)层对训练深层网络很重要。VGG-19结构类似于VGG-16，有略好于VGG-16的性能，但VGG-19需要消耗更大的资源，因此实际中VGG-16使用得更多。由于VGG-16网络结构十分简单，并且很适合迁移学习，因此至今VGG-16仍在广泛使用。VGG-16和VGG-19取名源自作者所处研究组名(Visual Geometry Group)。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0ibOqiawHpjRPZpbGxEBxzpnUsNxuGxGQtWOqiaAXX9m5Y8HwhG4Vibu6pg/640?wx_fmt=jpeg)

##### GoogLeNet 
5M参数，ILSVRC 2014的冠军网络。GoogLeNet试图回答在设计网络时究竟应该选多大尺寸的卷积、或者应该选汇合层。其提出了Inception模块，同时用1×1、3×3、5×5卷积和3×3汇合，并保留所有结果。网络基本架构为：conv1 (64) -> pool1 -> conv2^2 (64, 192) -> pool2 -> inc3 (256, 480) -> pool3 -> inc4^5 (512, 512, 512, 528, 832) -> pool4 -> inc5^2 (832, 1024) -> pool5 -> fc (1000)。GoogLeNet的关键点是：(1). 多分支分别处理，并级联结果。(2). 为了降低计算量，用了1×1卷积降维。GoogLeNet使用了全局平均汇合替代全连接层，使网络参数大幅减少。GoogLeNet取名源自作者所处单位(Google)，其中L大写是为了向LeNet致敬，而Inception的名字来源于盗梦空间中的"we need to go deeper"梗。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0EKARUFLbSDsI9GSlPBwtSGibKxvyKbFRGgsalqJaGovj5lFgtGExhyg/640?wx_fmt=jpeg)

##### Inception v3/v4 

在GoogLeNet的基础上进一步降低参数。其和GoogLeNet有相似的Inception模块，但将7×7和5×5卷积分解成若干等效3×3卷积，并在网络中后部分把3×3卷积分解为1×3和3×1卷积。这使得在相似的网络参数下网络可以部署到42层。此外，Inception v3使用了批量归一层。Inception v3是GoogLeNet计算量的2.5倍，而错误率较后者下降了3%。Inception v4在Inception模块基础上结合了residual模块(见下文)，进一步降低了0.4%的错误率。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0WC2iblOU7slw5rTibI7SQ8Af1Cpymia0TMfUyF4KobvF9AE7ueRzWFdeg/640?wx_fmt=jpeg)


##### ResNet 

ILSVRC 2015的冠军网络。ResNet旨在解决网络加深后训练难度增大的现象。其提出了residual模块，包含两个3×3卷积和一个短路连接(左图)。短路连接可以有效缓解反向传播时由于深度过深导致的梯度消失现象，这使得网络加深之后性能不会变差。短路连接是深度学习又一重要思想，除计算机视觉外，短路连接也被用到了机器翻译、语音识别/合成领域。此外，具有短路连接的ResNet可以看作是许多不同深度而共享参数的网络的集成，网络数目随层数指数增加。ResNet的关键点是：(1). 使用短路连接，使训练深层网络更容易，并且重复堆叠相同的模块组合。(2). ResNet大量使用了批量归一层。(3). 对于很深的网络(超过50层)，ResNet使用了更高效的瓶颈(bottleneck)结构(下图右)。ResNet在ImageNet上取得了超过人的准确率。
![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia06uMQRoNZyfBPpnry9EUl543BwmNLeptibQjU1BzZsWjBxxpYICTiapiaQ/640?wx_fmt=jpeg)


##### preResNet
 ResNet的改进。preResNet整了residual模块中各层的顺序。相比经典residual模块(a)，(b)将BN共享会更加影响信息的短路传播，使网络更难训练、性能也更差；(c)直接将ReLU移到BN后会使该分支的输出始终非负，使网络表示能力下降；(d)将ReLU提前解决了(e)的非负问题，但ReLU无法享受BN的效果；(e)将ReLU和BN都提前解决了(d)的问题。preResNet的短路连接(e)能更加直接的传递信息，进而取得了比ResNet更好的性能。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0zrA17ibpCR3ycZ13NXNiaCsaANOu66Vn8p3nPPwaHgfz3h1EPu5icweiaQ/640?wx_fmt=jpeg)



##### ResNeXt
ResNet的另一改进。传统的方法通常是靠加深或加宽网络来提升性能，但计算开销也会随之增加。ResNeXt旨在不改变模型复杂度的情况下提升性能。受精简而高效的Inception模块启发，ResNeXt将ResNet中非短路那一分支变为多个分支。和Inception不同的是，每个分支的结构都相同。ResNeXt的关键点是：(1). 沿用ResNet的短路连接，并且重复堆叠相同的模块组合。(2). 多分支分别处理。(3). 使用1×1卷积降低计算量。其综合了ResNet和Inception的优点。此外，ResNeXt巧妙地利用分组卷积进行实现。ResNeXt发现，增加分支数是比加深或加宽更有效地提升网络性能的方式。ResNeXt的命名旨在说明这是下一代(next)的ResNet。


![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0aweVJyXv6T9hsC8Twm8Bm8eQBER4ic1YxheAib7Yu3Z0shvldSPatLRg/640?wx_fmt=jpeg)



##### 随机深度 
ResNet的改进。旨在缓解梯度消失和加速训练。类似于随机失活(dropout)，其以一定概率随机将residual模块失活。失活的模块直接由短路分支输出，而不经过有参数的分支。在测试时，前馈经过全部模块。随机深度说明residual模块是有信息冗余的。

 
![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0gGibXJslEv0aeGH6YyMriaicJAasdfNKvdsJFzyIKiaBEzQdqdmovBgKrg/640?wx_fmt=jpeg)





##### DenseNet 
其目的也是避免梯度消失。和residual模块不同，dense模块中任意两层之间均有短路连接。也就是说，每一层的输入通过级联(concatenation)包含了之前所有层的结果，即包含由低到高所有层次的特征。和之前方法不同的是，DenseNet中卷积层的滤波器数很少。DenseNet只用ResNet一半的参数即可达到ResNet的性能。实现方面，作者在大会报告指出，直接将输出级联会占用很大GPU存储。后来，通过共享存储，可以在相同的GPU存储资源下训练更深的DenseNet。但由于有些中间结果需要重复计算，该实现会增加训练时间。

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0WWSXzR2Hu1XAFmibYgicOy6dTFuA6MFtPvCwYYakt81U0fJJicQ7O7PsQ/640?wx_fmt=jpeg)

##### 下表对比了上述几种网络结构

![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0pBlb09ricqmuoiaqNyUianLicfbM5EbpHEgQxkLZc4iaPG4CmJDiaWO8hsZA/640?wx_fmt=jpeg)

### 2. Object detection

#### 基本思路(滑窗)

多任务学习，网络带有两个输出分支。一个分支用于做图像分类，即全连接+softmax判断目标类别，和单纯图像分类区别在于这里还另外需要一个“背景”类。另一个分支用于判断目标位置，即完成回归任务输出四个数字标记包围盒位置(例如中心点横纵坐标和包围盒长宽)，该分支输出结果只有在分类分支判断不为“背景”时才使用。

基于滑动窗的目标检测动机是，尽管原图中可能包含多个目标，但滑动窗对应的图像局部区域内通常只会有一个目标(或没有)。因此，我们可以沿用目标定位的思路对窗口内区域逐个进行处理。但是，由于该方法要把图像所有区域都滑动一遍，而且滑动窗大小不一，这会带来很大的计算开销。

***为了解决这一问题，有两条路线：R-CNN(基于候选区域) /YOLO（基于直接回归）***

#### 基于候选区域

对一副图中的所有类别框出其所在位置，类别数不确定，所以输出数量也不确定，所以可以把检测也看成一个分类问题，而不是回归问题。对一副图我们需要不同尺度的框去探测各个位置可能出现的类别信息，最简单的方法就是try them all!但是这回带来巨大的运算量。以前人们用HOG特征来进行检测时是可以实现的，因为提取HOG特征很快（是用线性分类器），现在用CNN来提特征就使得全试一遍这种方法很不现实。于是人们提出各种方法来预先得到一些类别框可能在的位置，得到可能位置的一个小子集，以减少运算量。
这种预先提出候选区域的方法叫Region Proposals
这种方法的大致原理就是寻找相似像素块，把临近相似的像素区域框出来，这样做运算很快，但得到的框不是很准确，但主要目的还是达到了，就是排除很多无关的框的位置，大大减少了之后的运算量。在Region Proposal方法中最有名的的就是Selective search  ——[【Uijlings et al, “Selective Search for Object Recognition”, IJCV 2013】](https://blog.csdn.net/mao_kun/article/details/50576003)

![](https://img-blog.csdn.net/20170517134757881)

##### R-CNN 

先利用一些非深度学习的类别无关的无监督方法，在图像中找到一些可能包含目标的候选区域。之后，对每个候选区域前馈网络，进行目标定位，即两分支(分类+回归)输出。其中，我们仍然需要回归分支的原因是，候选区域只是对包含目标区域的一个粗略的估计，我们需要有监督地利用回归分支得到更精确的包围盒预测结果。R-CNN的重要性在于当时目标检测已接近瓶颈期，而R-CNN利于在ImageNet预训练模型微调的方法一举将VOC上mAP由35.1%提升至53.7%，确定了深度学习下目标检测的基本思路。一个有趣之处是R-CNN论文开篇第一句只有两个词"Features matter." 这点明了深度学习方法的核心。


候选区域(region proposal) 

候选区域生成算法通常基于图像的颜色、纹理、面积、位置等合并相似的像素，最终可以得到一系列的候选矩阵区域。这些算法，如selective search或EdgeBoxes，通常只需要几秒的CPU时间，而且，一个典型的候选区域数目是2k，相比于用滑动窗把图像所有区域都滑动一遍，基于候选区域的方法十分高效。另一方面，这些候选区域生成算法的查准率(precision)一般，但查全率(recall)通常比较高，这使得我们不容易遗漏图像中的目标。

![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia04OhClavYyuR8V0IhJbfkYB8f2TGa03dDNlWI7FEeHAoPPbk0eCiaugg/640?wx_fmt=jpeg)

##### Fast R-CNN 

R-CNN的弊端是需要多次前馈网络，这使得R-CNN的运行效率不高，预测一张图像需要47秒。Fast R-CNN同样基于候选区域进行目标检测，但受SPPNet启发，在Fast R-CNN中，不同候选区域的卷积特征提取部分是共享的。也就是说，我们先将整副图像前馈网络，并提取conv5卷积特征。之后，基于候选区域生成算法的结果在卷积特征上进行采样，这一步称为兴趣区域汇合。最后，对每个候选区域，进行目标定位，即两分支(分类+回归)输出

***兴趣区域汇合(region of interest pooling, RoI pooling)***

兴趣区域汇合旨在由任意大小的候选区域对应的局部卷积特征提取得到固定大小的特征，这是因为下一步的两分支网络由于有全连接层，需要其输入大小固定。其做法是，先将候选区域投影到卷积特征上，再把对应的卷积特征区域空间上划分成固定数目的网格(数目根据下一步网络希望的输入大小确定，例如VGGNet需要7×7的网格)，最后在每个小的网格区域内进行最大汇合，以得到固定大小的汇合结果。和经典最大汇合一致，每个通道的兴趣区域汇合是独立的。



![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0AEtOFfmjkUibpvRqyJsBrkKfFCRZLuZqzgvkticxmLgCnz2TJbdnCnsg/640?wx_fmt=jpeg)

##### Faster R-CNN 

Fast R-CNN测试时每张图像前馈网络只需0.2秒，但瓶颈在于提取候选区域需要2秒。Faster R-CNN不再使用现有的无监督候选区域生成算法，而利用候选区域网络从conv5特征中产生候选区域，并且将候选区域网络集成到整个网络中端到端训练。Faster R-CNN的测试时间是0.2秒，接近实时。后来有研究发现，通过使用更少的候选区域，可以在性能损失不大的条件下进一步提速。



候选区域网络(region proposal networks, RPN) 在卷积特征上的通过两层卷积(3×3和1×1卷积)，输出两个分支。其中，一个分支用于判断每个锚盒是否包含了目标，另一个分支对每个锚盒输出候选区域的4个坐标。候选区域网络实际上延续了基于滑动窗进行目标定位的思路，不同之处在于候选区域网络在卷积特征而不是在原图上进行滑动。由于卷积特征的空间大小很小而感受野很大，即使使用3×3的滑动窗，也能对应于很大的原图区域。Faster R-CNN实际使用了3组大小(128×128、256×256、512×512)、3组长宽比(1:1、1:2、2:1)，共计9个锚盒，这里锚盒的大小已经超过conv5特征感受野的大小。对一张1000×600的图像，可以得到20k个锚盒。

为什么要使用锚盒(anchor box) 

锚盒是预先定义形状和大小的包围盒。使用锚盒的原因包括：(1). 图像中的候选区域大小和长宽比不同，直接回归比对锚盒坐标修正训练起来更困难。(2). conv5特征感受野很大，很可能该感受野内包含了不止一个目标，使用多个锚盒可以同时对感受野内出现的多个目标进行预测。(3). 使用锚盒也可以认为这是向神经网络引入先验知识的一种方式。我们可以根据数据中包围盒通常出现的形状和大小设定一组锚盒。锚盒之间是独立的，不同的锚盒对应不同的目标，比如高瘦的锚盒对应于人，而矮胖的锚盒对应于车辆。

![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia09apoE4Yib23n3Dnpp9PFcibYaQhxotiaNjlGUIvDwbkHccXcaeMqT8Kkg/640?wx_fmt=jpeg)

##### R-FCN 

Faster R-CNN在RoI pooling之后，需要对每个候选区域单独进行两分支预测。R-FCN旨在使几乎所有的计算共享，以进一步加快速度。由于图像分类任务不关心目标具体在图像的位置，网络具有平移不变性。但目标检测中由于要回归出目标的位置，所以网络输出应当受目标平移的影响。为了缓和这两者的矛盾，R-FCN显式地给予深度卷积特征各通道以位置关系。在RoI汇合时，先将候选区域划分成3×3的网格，之后将不同网格对应于候选卷积特征的不同通道，最后每个网格分别进行平均汇合。R-FCN同样采用了两分支(分类+回归)输出。

![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0g302x55vXDZUMB6GhZvG27p1yJAiaMygDFVg7hmJlolaE7h9hZZIqRg/640?wx_fmt=jpeg)


##### 小结 

基于候选区域的目标检测算法通常需要两步：第一步是从图像中提取深度特征，第二步是对每个候选区域进行定位(包括分类和回归)。其中，第一步是图像级别计算，一张图像只需要前馈该部分网络一次，而第二步是区域级别计算，每个候选区域都分别需要前馈该部分网络一次。因此，第二步占用了整体主要的计算开销。R-CNN, Fast R-CNN, Faster R-CNN, R-FCN这些算法的演进思路是逐渐提高网络中图像级别计算的比例，同时降低区域级别计算的比例。R-CNN中几乎所有的计算都是区域级别计算，而R-FCN中几乎所有的计算都是图像级别计算。

[各算法在各数据集上的性能排名。](http://rodrigob.github.io/are_we_there_yet/build/)

#### 基于直接回归

***基本思路***

基于候选区域的方法由于有两步操作，虽然检测性能比较好，但速度上离实时仍有一些差距。基于直接回归的方法不需要候选区域，直接输出分类/回归结果。这类方法由于图像只需前馈网络一次，速度通常更快，可以达到实时。

##### YOLO 

将图像划分成7×7的网格，其中图像中的真实目标被其划分到目标中心所在的网格及其最接近的锚盒。对每个网格区域，网络需要预测：每个锚盒包含目标的概率(不包含目标时应为0，否则为锚盒和真实包围盒的IoU)、每个锚盒的4个坐标、该网格的类别概率分布。每个锚盒的类别概率分布等于每个锚盒包含目标的概率乘以该网格的类别概率分布。相比基于候选区域的方法，YOLO需要预测包含目标的概率的原因是，图像中大部分的区域不包含目标，而训练时只有目标存在时才对坐标和类别概率分布进行更新。



YOLO的优点在于：(1). 基于候选区域的方法的感受野是图像中的局部区域，而YOLO可以利用整张图像的信息。(2). 有更好的泛化能力。



YOLO的局限在于：(1). 不能很好处理网格中目标数超过预设固定值，或网格中有多个目标同时属于一个锚盒的情况。(2). 对小目标的检测能力不够好。(3). 对不常见长宽比的包围盒的检测能力不强。(4). 计算损失时没有考虑包围盒大小。大的包围盒中的小偏移和小的包围盒中的小偏移应有不同的影响。

![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0vrtM40EBqiaJaPXM60M053X8H3YUxlXt0gT4icE8D6QpLv2QDkq7vu9Q/640?wx_fmt=jpeg)

##### SSD 

相比YOLO，SSD在卷积特征后加了若干卷积层以减小特征空间大小，并通过综合多层卷积层的检测结果以检测不同大小的目标。此外，类似于Faster R-CNN的RPN，SSD使用3×3卷积取代了YOLO中的全连接层，以对不同大小和长宽比的锚盒来进行分类/回归。SSD取得了比YOLO更快，接近Faster R-CNN的检测性能。后来有研究发现，相比其他方法，SSD受基础模型性能的影响相对较小。
![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0E7PloJa5DNN1pReQVCF1iawLmZGM7VdaZqJEjPcqHDXxMxheNXRKmog/640?wx_fmt=jpeg)


##### FPN 

之前的方法都是取高层卷积特征。但由于高层特征会损失一些细节信息，FPN融合多层特征，以综合高层、低分辨率、强语义信息和低层、高分辨率、弱语义信息来增强网络对小目标的处理能力。此外，和通常用多层融合的结果做预测的方法不同，FPN在不同层独立进行预测。FPN既可以与基于候选区域的方法结合，也可以与基于直接回归的方法结合。FPN在和Faster R-CNN结合后，在基本不增加原有模型计算量的情况下，大幅提高对小目标的检测性能。
![pic](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0s6R1RJibTlf3jgcr8yuEAuyurmjaSAaACiaveXO5UIRppEQiaL9cQqsHw/640?wx_fmt=jpeg)

##### RetinaNet 

RetinaNet认为，基于直接回归的方法性能通常不如基于候选区域方法的原因是，前者会面临极端的类别不平衡现象。基于候选区域的方法可以通过候选区域过滤掉大部分的背景区域，但基于直接回归的方法需要直接面对类别不平衡。因此，RetinaNet通过改进经典的交叉熵损失以降低对已经分的很好的样例的损失值，提出了焦点(focal)损失函数，以使模型训练时更加关注到困难的样例上。RetinaNet取得了接近基于直接回归方法的速度，和超过基于候选区域的方法的性能。
![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0dyZfNjyhynicYF1xuWEqI6MLjR8chqtXWPXXVrpZ2KoG7ZM3icpL7qhQ/640?wx_fmt=jpeg)
![](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb0YWH76zPg7YBaDOsnYOzia0hvUCqj0Y2e2eMlfZUed8lTCx0pFFZqZr4tNhXAqlnZicwKY7fQ7MbEQ/640?wx_fmt=jpeg)

#### 目标检测常用技巧

##### 非最大抑制(non-max suppression, NMS) 

目标检测可能会出现的一个问题是，模型会对同一目标做出多次预测，得到多个包围盒。NMS旨在保留最接近真实包围盒的那一个预测结果，而抑制其他的预测结果。NMS的做法是，首先，对每个类别，NMS先统计每个预测结果输出的属于该类别概率，并将预测结果按该概率由高至低排序。其次，NMS认为对应概率很小的预测结果并没有找到目标，所以将其抑制。然后，NMS在剩余的预测结果中，找到对应概率最大的预测结果，将其输出，并抑制和该包围盒有很大重叠(如IoU大于0.3)的其他包围盒。重复上一步，直到所有的预测结果均被处理。

##### 在线困难样例挖掘(online hard example mining, OHEM) 

目标检测的另一个问题是类别不平衡，图像中大部分的区域是不包含目标的，而只有小部分区域包含目标。此外，不同目标的检测难度也有很大差异，绝大部分的目标很容易被检测到，而有一小部分目标却十分困难。OHEM和Boosting的思路类似，其根据损失值将所有候选区域进行排序，并选择损失值最高的一部分候选区域进行优化，使网络更关注于图像中更困难的目标。此外，为了避免选到相互重叠很大的候选区域，OHEM对候选区域根据损失值进行NMS。

##### 在对数空间回归 

回归相比分类优化难度大了很多。L2\ell_损失对异常值比较敏感，由于有平方，异常值会有大的损失值，同时会有很大的梯度，使训练时很容易发生梯度爆炸。而L1\el损失的梯度不连续。在对数空间中，由于数值的动态范围小了很多，回归训练起来也会容易很多。此外，也有人用平滑的L1\el损失进行优化。预先将回归目标规范化也会有助于训练。


##四. 参考文献
[1. 图像分类、检测，语义分割等方法梳理](https://yq.aliyun.com/articles/604579)
[2. 计算机视觉四大基本任务的应用知识分享](https://blog.csdn.net/u011707542/article/details/79151978)