# ConvNeXt-V2 pytorch 复现
[`训练Training..........`](https://github.com/Jacky-Android/convnext-v2-pytorch/tree/main#%E4%BB%A3%E7%A0%81%E4%BD%BF%E7%94%A8%E7%AE%80%E4%BB%8B)
# 2023年10月11日更新
## 加入FCMAE
全卷积掩码自编码器（FCMAE）框架是一种基于卷积神经网络的自监督学习方法，它的思想是在输入图像上随机掩盖一些区域，然后让模型尝试恢复被掩盖的部分。这样可以迫使模型学习到图像的全局和局部特征，从而提高其泛化能力。

FCMAE 框架与传统的掩码自编码器（MAE）框架相比，有两个优势：一是它使用了全卷积结构，而不是使用全连接层来生成掩码和重建图像，这样可以减少参数量和计算量，同时保持空间信息；二是它使用了多尺度掩码策略，而不是使用固定大小的掩码，这样可以增加模型对不同尺度特征的感知能力。
[FCMAE(fully convolutional masked autoencoder framework)](https://github.com/Jacky-Android/convnext-v2-pytorch/blob/main/fcmae_model.py)

![image](https://github.com/Jacky-Android/convnext-v2-pytorch/assets/55181594/cb3f3944-c0b6-4bba-86b3-d38f75fadcc6)

输入tensor[1,3,224,224],返回loss,pred,mask

torch.Size([]) torch.Size([1, 3072, 7, 7]) torch.Size([1, 49])
## FCMAE训练文件不定期更新
### torchinfo输出代码
```python
from fcmae_model import convnextv2_pico
from torchinfo import summary
import torch

#use pico
models = convnextv2_pico().cuda()
x = torch.randn([1,3,224,224]).cuda()
print(models(x)[0],models(x)[1].shape,models(x)[2].shape)
out = summary(models, (1, 3, 224,224))
```
### 模型torchinfo输出
```python
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
FCMAE                                         --                        512
├─SparseConvNeXtV2: 1-1                       [1, 512, 7, 7]            --
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-1                   [1, 64, 56, 56]           3,264
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-2                   [1, 64, 56, 56]           73,856
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-3                   [1, 128, 28, 28]          33,024
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-4                   [1, 128, 28, 28]          278,784
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-5                   [1, 256, 14, 14]          131,584
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-6                   [1, 256, 14, 14]          3,245,568
│    └─ModuleList: 2-7                        --                        (recursive)
│    │    └─Sequential: 3-7                   [1, 512, 7, 7]            525,312
│    └─ModuleList: 2-8                        --                        (recursive)
│    │    └─Sequential: 3-8                   [1, 512, 7, 7]            4,260,864
├─Conv2d: 1-2                                 [1, 512, 7, 7]            262,656
├─Sequential: 1-3                             [1, 512, 7, 7]            --
│    └─Block: 2-9                             [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-9                       [1, 512, 7, 7]            25,600
│    │    └─LayerNorm: 3-10                   [1, 7, 7, 512]            1,024
│    │    └─Linear: 3-11                      [1, 7, 7, 2048]           1,050,624
│    │    └─GELU: 3-12                        [1, 7, 7, 2048]           --
│    │    └─GRN: 3-13                         [1, 7, 7, 2048]           4,096
│    │    └─Linear: 3-14                      [1, 7, 7, 512]            1,049,088
│    │    └─Identity: 3-15                    [1, 512, 7, 7]            --
├─Conv2d: 1-4                                 [1, 3072, 7, 7]           1,575,936
===============================================================================================
Total params: 12,521,792
Trainable params: 12,521,792
Non-trainable params: 0
Total mult-adds (M): 235.88
===============================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 94.93
Params size (MB): 50.09
Estimated Total Size (MB): 145.62
===============================================================================================
```
## 代码使用简介
[参考代码](https://github.com/facebookresearch/ConvNeXt-V2)

[论文ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)


1. 下载好数据集，代码中默认使用的是花分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz),
如果下载不了的话可以通过kaggle链接下载: https://www.kaggle.com/datasets/l3llff/flowers
2. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
3. 下载预训练权重，在`model.py`文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重
4. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
5. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
6. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
7. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片的文件夹绝对路径，最后生成results.csv
8. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
9. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数

## Results and Pre-trained Models
### ImageNet-1K FCMAE pre-trained weights (*self-supervised*)
| name | resolution | #params | model |
|:---:|:---:|:---:|:---:|
| ConvNeXt V2-A | 224x224 | 3.7M | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt) |
| ConvNeXt V2-F | 224x224 | 5.2M | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt) |
| ConvNeXt V2-P | 224x224 | 9.1M | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt) |
| ConvNeXt V2-N | 224x224 | 15.6M| [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt) |
| ConvNeXt V2-T | 224x224 | 28.6M| [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt) |
| ConvNeXt V2-B | 224x224 | 89M  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt) |
| ConvNeXt V2-L | 224x224 | 198M | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt) |
| ConvNeXt V2-H | 224x224 | 660M | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt) |

### ImageNet-1K fine-tuned models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt V2-A | 224x224 | 76.7 | 3.7M  | 0.55G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt) |
| ConvNeXt V2-F | 224x224 | 78.5 | 5.2M  | 0.78G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt) |
| ConvNeXt V2-P | 224x224 | 80.3 | 9.1M  | 1.37G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt) |
| ConvNeXt V2-N | 224x224 | 81.9 | 15.6M | 2.45G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt) |
| ConvNeXt V2-T | 224x224 | 83.0 | 28.6M | 4.47G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt) |
| ConvNeXt V2-B | 224x224 | 84.9 | 89M   | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt) |
| ConvNeXt V2-L | 224x224 | 85.8 | 198M  | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt) |
| ConvNeXt V2-H | 224x224 | 86.3 | 660M  | 115G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt) |

### ImageNet-22K fine-tuned models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:| :---:|
| ConvNeXt V2-N | 224x224 | 82.1 | 15.6M | 2.45G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt)|
| ConvNeXt V2-N | 384x384 | 83.4 | 15.6M | 7.21G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt)|
| ConvNeXt V2-T | 224x224 | 83.9 | 28.6M | 4.47G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt)|
| ConvNeXt V2-T | 384x384 | 85.1 | 28.6M | 13.1G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt)|
| ConvNeXt V2-B | 224x224 | 86.8 | 89M   | 15.4G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt)|
| ConvNeXt V2-B | 384x384 | 87.7 | 89M   | 45.2G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt)|
| ConvNeXt V2-L | 224x224 | 87.3 | 198M  | 34.4G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt)|
| ConvNeXt V2-L | 384x384 | 88.2 | 198M  | 101.1G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt)|
| ConvNeXt V2-H | 384x384 | 88.7 | 660M  | 337.9G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt)|
| ConvNeXt V2-H | 512x512 | 88.9 | 660M  | 600.8G  | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt)|
