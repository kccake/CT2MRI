# 使用 编码器、解码器的思路 做 CT 转 MRI

具体而言，参考stylegan2与U-net

利用U-net的编码器生成 CT在隐变量空间中的映射（stylegan的MappingNetwork）

利用U-net的解码器做 GAN的生成器（stylegan的generator）

利用stylegan的判别器做 GAN的判别器

## 注意事项

由于换了显卡平台，config中的目录做了更改

## 更新日志

### 2025-03-25

创建仓库，留下基本的加载数据等模块，models亟需实现

