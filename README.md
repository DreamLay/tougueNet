# 基于U-net深度学习细胞纹路检测语义分割


- 此项目模型基于医学上著名的老牌网络unet开发。
- 由于无法收集到舌头裂纹数据集，故暂且使用细胞图。
- 个人认为细胞图检测分要简单于舌头裂纹检测。
- 网上对比了下正常与裂纹的舌头图片，正常舌头自然舒展不像手掌心本来就有纹路而舌头没有，所以舌头上的皱泽应该不会对裂纹检测造成太大的干扰。


### 运行：
    python predict.py path（python main.py /home/.../.../1.png）识别并显示标记的图片。
    macos系统可以正常显示标记图片，理论linux也可以可视化

### 效果：
    尽管只有30张图片，但通过数据增强，模型已训练至准确率91.81%。

### 环境：

- system: MacOS Majave/Ubuntu18.04
- packs: tensorflow, numpy, PLI, keras, cv2, h5py

![avatar](https://github.com/DreamLay/tougueNet/blob/master/example/train-volume.jpg)
![avatar](https://github.com/DreamLay/tougueNet/blob/master/example/train-labels.jpg)
