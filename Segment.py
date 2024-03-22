import numpy as np
import cv2
def show(img):
    if img.ndim ==3:
        h,w,_ = img.shape
    else:
        h, w = img.shape
    img = cv2.resize(img,(w//3,h//3))
    cv2.imshow("kmeans", img)
    cv2.waitKey()

def segGray(roi1,roi2,roi3,thred = 20):
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
    _, segment1 = cv2.threshold(gray1, thred, 255, cv2.THRESH_BINARY)
    segment1 = segment1.reshape(segment1.shape[0],segment1.shape[0],1)
    segment1 = np.concatenate( [segment1,segment1,segment1] ,axis = 2)

    _, segment2 = cv2.threshold(gray2, thred, 255, cv2.THRESH_BINARY)
    segment2 = segment2.reshape(segment2.shape[0], segment2.shape[0], 1)
    segment2 = np.concatenate([segment2, segment2, segment2], axis=2)

    _, segment3 = cv2.threshold(gray3, thred, 255, cv2.THRESH_BINARY)
    segment3 = segment3.reshape(segment3.shape[0], segment3.shape[0], 1)
    segment3 = np.concatenate([segment3, segment3, segment3], axis=2)
    return segment1,segment2,segment3


def segKmeans(roi1,roi2,roi3,iterations):
    return Kmeans(roi1,iterations), Kmeans(roi2,iterations), Kmeans(roi3,iterations)

def Kmeans(img,iterations):
    des = [27,26,37]#箭支RGB特征
    des = np.uint8(des)
    reshaped_image = img.reshape(-1, 3).astype(np.float32)
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    _, labels, centers = cv2.kmeans(reshaped_image, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    for each in range(0,5):
        if np.linalg.norm(centers[each] - des) < 40:
            centers[each] = [255]
        else:
            centers[each] = [0]

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image


