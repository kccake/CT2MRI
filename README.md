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

