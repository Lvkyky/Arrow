# 射箭自动判分算法

![image](https://github.com/Lvkyky/Arrow/assets/87217038/94142bc2-4384-41b2-a202-f06b6d1d0efd)

# 描述
  算法利用RGB图像基于多视角策略对箭支的落点进行检测以实现自动判分，首先利用二维码和ORB特征对来自于不同视角的四副图像进行对齐。其次，利用无监督图像分割算法（Kmeans）将箭支分割出来。然后，利用膨胀和辐射操作去掉圆弧噪声，利用面积滤除穿孔噪声。最后，求取箭支的重叠区域。计算重叠区域的几何中心即为最终的落点。
算法以外的改进建议：
  0：仔细调整机位尽量消除旋转误差。
  1：多贴一些二维码提高对齐算法精度。
  2：调整光线使得箭支反光减少以增大箭支与背景之间的差异提高分割算法精度。
  3：将相机靠近靶面或调节镜头增加检测区域的视野。

