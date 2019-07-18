# 1. 训练好的模型推理图片
用infer_simple.py文件

# 2. 训练自己的图片


## 2.1 COCO数据集训练

第一步： 
下载coco数据集：coco数据官网（我下的是2014），在data下新建文件夹命名为coco，把下载的数据移动到coco下： 

![](https://img-blog.csdn.net/20180626142246316?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3l1NzM0MzkwODUz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

第二步： 
修改dataset_catalog.py文件里的路径（默认的不懂也行），如下： 
 ![](https://img-blog.csdn.net/20180626142608295?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3l1NzM0MzkwODUz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
第三步： 
在 $/detectron/configs/getting_started 下新建空白文档（我命名为：tutorial_1gpu_e2e_mask_rcnn_R-50-FPN.yaml），仿照tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml和e2e_mask_rcnn_R-50-FPN_2x.yaml 完成mask_rcnn配置文件。最终我的配置文件如下：https://download.csdn.net/download/yu734390853/10500353 

第四步：训练
```
python2 tools/train_net.py \
    --cfg configs/getting_started/tutorial_1gpu_e2e_mask_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output
```
测试：
```

python2 tools/infer_simple.py \
    --cfg configs/getting_started/tutorial_1gpu_e2e_mask_rcnn_R-50-FPN.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts /home/ymh/detectron/model/model_final.pkl \
    demo
```


注：其中的model_final.pkl是自己训练时保存的最后一个模型文件。



## 2.2 Labelme安装与使用 

### 安装：
sudo apt-get install python-pyqt5  # PyQt5
sudo pip install labelme
### 使用：

labelme

## 2.3 Labelme转coco

使用datasetPreprocessing/labelme2COCO.py 文件

---datasetPreprocessing
---------labelme2COCO.py（将lebelme标注产生的数据转换为coco所需数据集格式代码）
---------data_process_config.json（数据转换配置文件）

data_process_config.json（配置文件）

```
"labelme2cocoParams":{ "labelme_dataset_dir" : "pre_ds_log_2018_09_14", #使用labelme标注产生的数据集（需要放在主目录下/root/data/labelme文件夹下）

"coco_dataset_dir" : "coco_data" # 适用于coco训练的数据集名称(会在/root/data/coco目录下生成) }

```

labelme2COCO.py

```
python labelme2COCO.py

```

## 2.4 Faster-Rcnn训练

 
### 2.4.1 配置训练参数 

修改config_mrcnn/12_2017_baselines/下的yaml文件
---detectron 
    -----config_mrcnn 
        --------12_2017_baselines(yaml配置文件目录，配置训练参数) 
---configs（该目录下有许多原作者训练时保存的配置文件） 
(可以复制原有的yaml文件，也可以在configs/下选择其他的yaml文件cope过来，将命名修改为e2e_mask_rcnn_R-50-FPN_1x_for_objectname.yam)

```
MODEL: 
TYPE: generalized_rcnn 
CONV_BODY: FPN.add_fpn_ResNet50_conv5_body 
NUM_CLASSES: 3 //此处为物体类别+1
FASTER_RCNN: True 
MASK_ON: True 
NUM_GPUS: 1 //GPU个数 
SOLVER: 
WEIGHT_DECAY: 0.0001 
LR_POLICY: steps_with_decay 
BASE_LR: 0.002 
GAMMA: 0.1
MAX_ITER: 90000 //迭代次数 
STEPS: [0, 60000, 80000] 
FPN: 
FPN_ON: True 
MULTILEVEL_ROIS: True 
MULTILEVEL_RPN: True 
FAST_RCNN: 
ROI_BOX_HEAD: fast_rcnn_heads.add_roi_2mlp_head 
ROI_XFORM_METHOD: RoIAlign 
ROI_XFORM_RESOLUTION: 7 
ROI_XFORM_SAMPLING_RATIO: 2 
MRCNN: 
ROI_MASK_HEAD: mask_rcnn_heads.mask_rcnn_fcn_head_v1up4convs 
RESOLUTION: 28 # (output mask resolution) default 14 
ROI_XFORM_METHOD: RoIAlign 
ROI_XFORM_RESOLUTION: 14 # default 7 
ROI_XFORM_SAMPLING_RATIO: 2 # default 0 
DILATION: 1 # default 2 
CONV_INIT: MSRAFill # default GaussianFill 
TRAIN: 
WEIGHTS: https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl //初始权重，可以指定为本地pkl
DATASETS: ('aowh_ha_train',) //训练数据集名称，与前面的dataset_catalog.py中名字保持一致 
SCALES: (800,) 
MAX_SIZE: 1500 
BATCH_SIZE_PER_IM: 64 
RPN_PRE_NMS_TOP_N: 2000 # Per FPN level 
TEST: 
DATASETS: ('aowh_ha_val',) //验证数据集名称，与前面的dataset_catalog.py中名字保持一致 
SCALE: 800 
MAX_SIZE: 1333 
NMS: 0.5 
RPN_PRE_NMS_TOP_N: 1000 # Per FPN level 
RPN_POST_NMS_TOP_N: 1000 
OUTPUT_DIR: .
```

### 2.4.2 执行训练

```
python tools/train_net.py 
--cfg /root/detectron/configs/getting_started/lyf_tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml\  //之前配置的yaml文件 
OUTPUT_DIR /root/detectron/models/   //模型保存地址
```



# 3. 模型导出与调用 
## 3.1 pkl to pb
detectron训练出来的目标检测模型后缀为.pkl，这种模型在使用的时候必须要有图（graph）以及detectron代码的支持，转为caffe2标准的pb模型后，就可以脱离detectron的代码单独运行，非常方便。（经过翻阅github的相关资料，转为pb模型后可以支持在c++上运行？）

转换所要用到的函数为tools/convert_pkl_to_pb.py。

进入这个函数，主要要填写的部分为，以下**黑体**的部分是必须要修改的，其余是可以按需求修改

***1.cfg***
    parser.add_argument(
        ***'--cfg', dest='cfg_file', help='optional config file', default=None,        type=str)***
即cfg的路径。此外cfg文件中的TEST部分添加：

TEST:
  WEIGHTS:#  你的pkl文件路径

2.test_img
test_img是测试转换后的pb文件与原始的pkl文件是否一致，有没有出错。如果添加了test_img，就会进行对比测试，如果没有添加，demo就不会检查，只会输出pb模型文件

    parser.add_argument(
        '--test_img', dest='test_img',
        help='optional test image, used to verify the model conversion',
        default=None,
        type=str)
***3.使用设备（device）***
    parser.add_argument(
        '--device', dest='device',
        help='Device to run the model on',
        choices=['cpu', 'gpu'],
        ***default='cpu',***
        type=str)
如果你准备在cpu上运行pb文件，那么选择cpy，否则选择gpu

4.pb模型的保存路径
    parser.add_argument(
        '--out_dir', dest='out_dir', help='output dir', default=None,
        type=str)

5.pb模型的名称
    parser.add_argument(
        '--net_name', dest='net_name', help='optional name for the net',
        default="detectron", type=str)
 

修改完上述所有之后，就可以运行啦，会在输出的路径中生成以下几个文件：

model.pb
model_init.pb
 

**可能出现的问题**

**A. fpn not support**

这个是老版本caffe2和detectron才会出现的情况，新版本的caffe2和detectron就已经做了修复。解决方法是：

（1）从官网上下载新版本的pytorch/modules/detectron，替换老版本的相应位置

（2）从官网上下载新版本的pytorch/caffe2/operators，替换老版本的相应位置

（3）从detectron官网上下载新版本的convert_pkl_to_pb.py，替换老版本。



**B. No module named caffe2.fb.predictor**

import路径不对,covert_pkl_to_pb的第44行：改成：

from caffe2.**python**.predictor import predictor_exporter, predictor_py_utils

**C. model = u, 模型导入为空：**

covert_pkl_to_pb的第451行：改成： model = test_engine.initialize_model_from_cfg(cfg.**TRAIN**.WEIGHTS)

**D. Error when writing graph to image [Errno 2] "dot" not found in path.**

缺少graphviz工具：
sudo apt-get install graphviz


## 3.2 模型调用

### 3.2.1 pkl模型python调用

#### 3.2.1.1 模型调用逻辑 

#### 3.2.1.2 数据处理逻辑

OPEN CV读取图片
```
infer_simple.py 第176行 im=cv2.imread(im_name)
                第184行 cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im, None, timers=timers)
```
把图片转成blobs
```
test.py 第52行  def im_detect_all(model, im, box_proposals, timers=None):
        第66行  scores, boxes, im_scale = im_detect_bbox(model, im， cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals)
        第943行 def _get_blobs(im, rois, target_scale, target_max_size):
                blobs['data'], im_scale, blobs['im_info'] = blob_utils.get_image_blob(im, target_scale, target_max_size)
```
图片预处理
```
blob.py 第51行  processed_im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_scale, target_max_size    )
                    return blob, im_scale, im_info.astype(np.float32)
```
缩放到指定尺寸
```
        第100行 def prep_im_for_blob(im, pixel_means, target_size, max_size):
                im = im.astype(np.float32, copy=False)
                im -= pixel_means
                im_shape = im.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(target_size) / float(im_size_min)
                # Prevent the biggest axis from being more than max_size
                if np.round(im_scale * im_size_max) > max_size:
                    im_scale = float(max_size) / float(im_size_max)
                im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR
                return im, im_scale
```
图片数据转成数组
```
        第54行 blob = im_list_to_blob(processed_im)
        第67行 def im_list_to_blob(ims):
                    """Convert a list of images into a network input. Assumes images were
                    prepared using prep_im_for_blob or equivalent: i.e.
                    - BGR channel order
                    - pixel means subtracted
                    - resized to the desired input size
                    - float32 numpy ndarray format
                    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
                    shape.
                    """
                    if not isinstance(ims, list):
                        ims = [ims]
                    max_shape = np.array([im.shape for im in ims]).max(axis=0)
                    # Pad the image so they can be divisible by a stride
                    if cfg.FPN.FPN_ON:
                        stride = float(cfg.FPN.COARSEST_STRIDE)
                        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
                        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

                    num_images = len(ims)
                    blob = np.zeros(
                        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32
                    )
                    for i in range(num_images):
                        im = ims[i]
                        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
                    # Move channels (axis 3) to axis 1
                    # Axis order will become: (batch elem, channel, height, width)
                    channel_swap = (0, 3, 1, 2)
                    blob = blob.transpose(channel_swap)
                    return blob
```

FeedBlob
```
test 第155行 for k, v in inputs.items():
             workspace.FeedBlob(core.ScopedName(k), v)
```
### 3.2.2 pb模型python 调用

https://www.wandouip.com/t5i70287/


#### 3.2.2.1 模型调用逻辑 
```
use_pb_model_with_python.py   161行  workspace.RunNet()

workspace.py 第251行 return  CallWithExceptionIntercept(
                                C.run_net,
                                C.Workspace.current._last_failed_op_net_position,
                                GetNetName(name),
                                StringifyNetName(name), num_iter, allow_fail,
                                )

workspace.py 第214行 def CallWithExceptionIntercept(C.run_net,.....）
                         return  C.run_net （）


workspace.py 第23行  import caffe2.python._import_c_extension as C

_import_c_extension.py 第27行 from caffe2.python.caffe2_pybind11_state_gpu import *  

```
#### 3.2.2.2 数据处理逻辑
```
use_pb_model_with_python.py 第179行 test_img = cv2.imread(test_img_file)
                            第191行  boxes, classids = run_model_pb(predict_net, init_net, test_img)
                            第118行 def run_model_pb(net, init_net, im):
                            第126行    input_blobs = _prepare_blobs(
                                                    im,
                                                    PIXEL_MEANS,
                                                    TEST_SCALE, TEST_MAX_SIZE
                                                      )
                            第78行 def _prepare_blobs(
                                                        im,
                                                        pixel_means,
                                                        target_size,
                                                        max_size,
                                                    ):
                                                    im = im.astype(np.float32, copy=False)
                                                    im -= pixel_means
                                                    im_shape = im.shape

                                                    im_size_min = np.min(im_shape[0:2])
                                                    im_size_max = np.max(im_shape[0:2])
                                                    im_scale = float(target_size) / float(im_size_min)
                                                    if np.round(im_scale * im_size_max) > max_size:
                                                        im_scale = float(max_size) / float(im_size_max)
                                                    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                                                    interpolation=cv2.INTER_LINEAR)

                                                    # Reuse code in blob_utils and fit FPN
                                                    blob = im_list_to_blob([im])

                                                    blobs = {}
                                                    blobs['data'] = blob
                                                    blobs['im_info'] = np.array(
                                                        [[blob.shape[2], blob.shape[3], im_scale]],
                                                        dtype=np.float32
                                                    )
                                                     return blobs
                            第133行     for k, v in input_blobs.items():
                                                    workspace.FeedBlob(
                                                        core.ScopedName(k),
                                                        v,
                                                        get_device_option_cuda() if k in gpu_blobs else
                                                        get_device_option_cpu()
                                                    )


```
#### 3.2.2.3 碰到的问题

1. Class类别无法显示：

vis.py文件中vis_one_image的输入数据类型不对应，原来pkl的模型输入的是cls_boxes包含了boxes和classes。现在把boxes和classes单独输入

2. GPU无法运行，也不报错：

pkl导出pb模型的时候类型选择需要选成gpu。


#### 3.2.2.4 python工程打包

安裝pyinstaller

pip install pyinstaller

复制 pycocotools：
cp -r /root/detectron/tools/build/use_pb_model_with_python/pycocotools-2.0-py2.7-linux-x86_64.egg/pycocotools /root/detectron/tools/dist/use_pb_model_with_python

matplot 必须是2.0.2以下：

pip uninstall matplotlib
pip install matplotlib==2.0.2

打包:
pyinstaller ***.py


## 3.2.3 pb模型C++调用


