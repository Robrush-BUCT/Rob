import cv2 as cv
import numpy as np
import math

camera = cv.VideoCapture(0)


color_range = {'yellow_door': [(15, 43, 46), (24, 255, 255)], 'black_door': [(0, 25, 0), (60, 255, 30)]}
Debug = True

def getAreaMaxContour(contours, area_min=36):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area_min:  # 只有在面积大于25时，最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c
    return area_max_contour, contour_area_max  # 返回最大的轮廓

while(1):
    ret, org_img = camera.read()
    t1 = cv.getTickCount()
    border = cv.copyMakeBorder(org_img, 12, 12, 16, 16, borderType=cv.BORDER_CONSTANT, value=(255, 255, 255))
    org_img_resize = cv.resize(border, (320, 240), interpolation=cv.INTER_CUBIC)
    frame_gauss = cv.GaussianBlur(org_img_resize, (3, 3), 0)
    frame_hsv = cv.cvtColor(frame_gauss, cv.COLOR_BGR2LAB)
    mark_yellow = cv.inRange(frame_hsv, color_range['yellow_door'][0], color_range['yellow_door'][1])
    mark_black = cv.inRange(frame_hsv, color_range['black_door'][0], color_range['black_door'][1])
    whole_mark = cv.add(mark_yellow, mark_black)
    opened = cv.morphologyEx(whole_mark, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
    percent = round(100 * area_max / (320 * 240), 2)  # 最大轮廓的百分比
    if areaMaxContour is not None:
        rect = cv.minAreaRect(areaMaxContour)  # 矩形框选
        box = np.int0(cv.boxPoints(rect))  # 点的坐标
        if Debug:
            cv.drawContours(org_img_resize, [box], 0, (153, 200, 0), 2)  # 将最小外接矩形画在图上
    if Debug:
        cv.imshow('closed', closed)  # 显示掩模
        cv.putText(org_img_resize, 'area: ' + str(percent) + '%', (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        t2 = cv.getTickCount()
        time_r = (t2 - t1) / cv.getTickFrequency()
        fps = 1.0 / time_r
        cv.putText(org_img_resize, "fps:" + str(int(fps)), (30, 200),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # cv.moveWindow('orgFrame', img_center_x, 100)  # 显示框位置
        cv.imshow('org_img_resize', org_img_resize)  # 显示图像
        cv.waitKey(1)

    if percent > 15:
        print(percent)
        time.sleep(0.01)
    else:
        print(percent)
        print("start to go")



