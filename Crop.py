import cv2
import numpy as np
from pyzbar import pyzbar
def avg(circles):
    sumx = 0
    sumy = 0
    sumr = 0
    sum = len(circles)
    for each in circles:
        sumx = sumx+each[0]/sum
        sumy  = sumy+each[1]/sum
        sumr = sumr+each[2]/sum
    return int(sumx),int(sumy),int(sumr)
def show(img):
    h,w,c = img.shape
    img = cv2.resize(img,(w//3,h//3))
    cv2.imshow("img", img)
    cv2.waitKey()



"""
用于裁剪图像，
返回感兴趣区域ROI1,ROI2,ROI3以及其中心以及半径
"""
#裁剪矩形区域
def getRectangle(srcImg, desImg):
        # barcodes= pyzbar.decode(imgGray[300:2700,180:3800])
        barcodes = pyzbar.decode(desImg)
        barcodesNew = [None, None, None, None]
        # 以上线霸争重新排序重新排序
        for barcode in barcodes:
            if barcode.data == b'\xe4\xb8\x8a':
                barcodesNew[0] = barcode
            elif barcode.data == b'\xe9\x83\xa4\xef\xbd\xbf':
                barcodesNew[1] = barcode
            elif barcode.data == b'\xe9\xab\xb4\xef\xbd\xb8':
                barcodesNew[2] = barcode
            elif barcode.data == b'\xe4\xba\x89':
                barcodesNew[3] = barcode

        if (len(barcodes) == 4):
            begain_x,begain_y,_,_ = barcodesNew[0].rect
            end_x,end_y,w,h = barcodesNew[3].rect
            end_x = end_x +w
            end_y = end_y + h
            return srcImg[begain_y:end_y, begain_x:end_x, :], desImg[begain_y:end_y, begain_x:end_x, :]

        else:
            # print("未检测到二维码,使用固定裁剪")
            begain_x, begain_y=0,0
            end_x, end_y=3800,2500
            return srcImg[begain_y:end_y , begain_x:end_x, :], desImg[begain_y:end_y,begain_x:end_x, :]
#裁剪感兴趣区域
def getRoi(img,param1=90,param2=40):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 进行圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=param1, param2=param2, minRadius=450,
                                   maxRadius=480)
        # 如果检测到了圆
        if circles is not None:
            pass
            # print("检测到圆")
        else:
            print("没检测到圆")
        # 将圆的坐标和半径转换为整数
        circles = np.round(circles[0, :]).astype("int")

        # 在图像上绘制检测到的圆
        circles1 = []
        circles2 = []
        circles3 = []
        for (x, y, r) in circles:
            if x < 1000:
                circles1.append([x, y, r])
            elif x > 1000 and x < 2000:
                circles2.append([x, y, r])
            else:
                circles3.append([x, y, r])
        result = []
        x, y, r = avg(circles1)
        result.append([x,y,r])
        result1 = img[y - r:y + r, x - r:x + r, :]
        # mask = np.zeros((w,w,3),dtype = np.uint8)
        # cv2.circle(mask,(r,r),r,(1,1,1),-1)
        # result1 = mask*result1
        # mask1= (1-mask)*255
        # result1 = result1+mask1
        # show(result1)

        x, y, r = avg(circles2)
        result.append([x, y, r])
        result2 = img[y - r:y + r, x - r:x + r, :]
        # mask = np.zeros((w, w, 3), dtype=np.uint8)
        # cv2.circle(mask, (r, r), r, (1, 1, 1), -1)
        # result2 = mask * result2
        # mask1 = (1 - mask) * 255
        # result2 = result2 + mask1

        x, y, r = avg(circles3)
        result.append([x, y, r])
        result3 = img[y - r:y + r, x - r:x + r, :]
        # mask = np.zeros((w, w, 3), dtype=np.uint8)
        # cv2.circle(mask, (r, r), r, (1, 1, 1), -1)
        # result3 = mask * result3
        # mask1 = (1 - mask) * 255
        # result3 = result3 + mask1
        return result1, result2, result3,result



