# CoDiff

1，这个工作是在上次发表的RoCo文章的基础之上去尝试用diffusion解决协同感知中的噪声问题（位姿误差和时间延迟），正在审稿中

2，记录训练过程，防止遗忘：
https://lx2xygwjrgr.feishu.cn/docx/AgrcdAIuOoIR9zxdozUcpXPZndh

3，主要修改的文件就是两个，一个是'point_pillar_baseline_multiscale.py'还有一个是'diffusion_fuse.py'，其中`diffusion_fuse.py`的内容是，将单个车辆的特征当作条件来指导diffusion生成总特征的过程

4，由于不同数据集需要的feature map大小不同，整个代码写的比较乱，只有我能看懂。。。。
