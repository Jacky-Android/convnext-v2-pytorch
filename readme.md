# ConvNeXt-V2 pytorch 复现

## 代码使用简介
[代码来源](https://github.com/facebookresearch/ConvNeXt-V2)

[论文ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)


1. 下载好数据集，代码中默认使用的是花分类数据集，下载地址: [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz),
如果下载不了的话可以通过百度云链接下载: https://pan.baidu.com/s/1QLCTA4sXnQAw_yvxPj9szg 提取码:58p0
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
