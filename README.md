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


