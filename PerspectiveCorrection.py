
import numpy as np
import cv2
from pyzbar import pyzbar

"""
用于矫正视角偏差，使得四个视角图像对齐
"""
def getQR(imgGray):
        barcodes= pyzbar.decode(imgGray[300:2700,180:3800])
        # barcodes = pyzbar.decode(imgGray)
        # print(len(barcodes))
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

        pointsList = []
        for barcode in barcodesNew:
            points = []
            if barcode != None:
                x, y, w, h = barcode.rect
                points.append([x, y])
                points.append([x + w, y])
                points.append([x + w, y + h])
                points.append([x, y + h])
                pointsList.append([barcode.data, points])
        return pointsList
#匹配关键点
def match(srcPoints,desPoints):
        Src = []
        Des = []
        for eachDes in desPoints:
            if eachDes != None:
                for eachSrc in srcPoints:
                    if eachSrc != None:
                        if eachSrc[0] == eachDes[0]:
                            Src.append(eachSrc[1])
                            Des.append(eachDes[1])
        return np.array(Src,dtype=np.float32), np.array(Des,dtype=np.float32)
#检测关键点
def getPoints(srcImg, desImg):
    srcList = getQR(cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY))
    desList = getQR(cv2.cvtColor(desImg, cv2.COLOR_BGR2GRAY))
    srcPoints, desPoints = match(srcList, desList)
    return srcPoints, desPoints

    #二维码矫正
#使用二维码进行对齐
def CorrectionByQR(srcImg, desImag, thre = 7.0):
        srcPoints,desPoints = getPoints(srcImg, desImag)
        batcn,num1,num2 = srcPoints.shape
        srcPoints = srcPoints.reshape((batcn*num1),num2)
        desPoints = desPoints.reshape((batcn*num1),num2)

        M, mask = cv2.findHomography(srcPoints, desPoints, cv2.RANSAC, thre)
        result = cv2.warpPerspective(srcImg, M, (desImag.shape[1], desImag.shape[0]))
        return result

        #特征点校正
#使用ORB特征进行对其
def CorrectionByFeature(srcImg, desImag, thred=5.0):

        # 创建SIFT对象
        orb = cv2.ORB_create()
        # 寻找关键点和计算描述符
        keypoints1, descriptors1 = orb.detectAndCompute(srcImg, None)
        keypoints2, descriptors2 = orb.detectAndCompute(desImag, None)
        # 创建 Brute-Force 匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 匹配描述符
        matches = bf.match(descriptors1, descriptors2)
        # 按照距离进行排序
        matches = sorted(matches, key=lambda x: x.distance)
        # 选择前N个最佳匹配结果
        N = 400
        good_matches = matches[:N]

        # 提取匹配结果的特征点坐标
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # 获取匹配结果中的特征点坐标
        M, mask = cv2.findHomography(points1, points2, cv2.RANSAC, thred)
        result = cv2.warpPerspective(srcImg, M, (desImag.shape[1], desImag.shape[0]))

        return result




