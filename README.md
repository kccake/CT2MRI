# CT2MR remaster

This is a remaster of the CT2MR dataset, which is a dataset of 3D CT and MR images of the head and neck region. The dataset is used for the task of multimodal image registration. 

The original dataset was created by the **West China Hospital of Sichuan University**, which means we cannot provide it without authorization. 

The remaster is created by the [XingYueChenFu](https://github.com/XingYueChenFu)

## Get Start
1. Please put the data in the `data` folder

    将数据放在`data`文件夹下

2. Check the `config.yaml` file for the path under `data`

    检查`config.yaml`文件的`data`下的路径等


3. Run `python main.py`

    运行`python main.py`

4. 可能需要安装的包

- torch torchvision

- pyyaml

- scikit-image

- tqdm

- pillow

- opencv-python

## Update log

### 2025-3-10

初步排除dataset加载中的bug

将starGAN模型几乎照搬到了该分支

### 2025-3-11

重塑配置方法

加入bugfree私货(可去除)

将大部分功能copy过来

修复只能将batch_size设置为1的bug：

  修复方法：将体积数据的depth用0填充为配置的max_depth，上下填充

修复tqdm进度条的bug

精简部分代码，修复一些现在记不清的东西

修复读取数据，将ct与mr混为一谈的**严重**bug

### 2025-3-12

加入NoneMedicalVolumePreprocessor类，用于不做对比度增强的数据预处理

    继承自MedicalVolumePreprocessor

### 2025-3-13

修复bugfree的一些bug，删掉冗余成分

### 2025-3-14

拆分 dataset加载 与 data processor

重塑data processor，将功能全部拆分，使其更加灵活（同时，如`main.py`等文件同步更改）

经验证可以正常运行
