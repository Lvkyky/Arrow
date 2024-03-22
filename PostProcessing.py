
import cv2
import numpy as np

def maskOuside(img,r):
    w = 2*r
    x = r
    y= r
    #对箭靶之外的二维码进行过滤
    maskOutside = np.zeros((w,w,3), dtype = np.uint8)
    cv2.circle(maskOutside,(x,y),int(r),(1,1,1,),-1)
    img = maskOutside*img
    return img

def getRing(seg1,seg2,seg3,se4,r):
    seg1 = maskOuside(seg1,r)/255
    seg2 = maskOuside(seg2,r)/255
    seg3 = maskOuside(seg3,r)/255
    seg4 = maskOuside(se4,r)/255
    print(seg1)

    result = seg1*seg2*seg3*seg4
    return result*255



def show(img):
    if img.ndim ==3:
        h,w,_ = img.shape
    else:
        h, w = img.shape
    img = cv2.resize(img,(w//3,h//3))
    cv2.imshow("img", img)
    cv2.waitKey()
def deCircle(x,y,r,img):
    x = r
    y = r
    w = 2*r
    #对箭靶之外的二维码进行过滤
    maskOutside = np.zeros((w,w,3), dtype = np.uint8)
    cv2.circle(maskOutside,(x,y),int(r),(1,1,1,),-1)
    img = maskOutside*img
    #将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #进行二值化处理，将图像转换为黑白图像
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 定义一个腐蚀核，指定腐蚀的形状和大小
    kernel = np.ones((4, 4), np.uint8)
    #进行腐蚀操作
    eroded = cv2.erode(binary, kernel, iterations=2)
    #进行膨胀操作
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    return dilated
def deNoising(img):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 定义面积阈值，过小的连通区域会被去除
    area_threshold = 1000

    # 迭代遍历轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积小于阈值，将轮廓对应的区域填充为背景色
        if area < area_threshold:
            cv2.drawContours(img, [contour], 0, (0, 0, 0), -1)

    #可选：使用形态学操作进一步优化结果
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return  binary
