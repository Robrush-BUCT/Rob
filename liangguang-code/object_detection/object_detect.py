import cv2
import numpy as np
import math


class OBJECT_DETECT():

    def __init__(self):
        # self.img = cv2.imread(filepath)
        self.color_hsv_threshold = {'yellow_door': [(10, 43, 46), (34, 255, 255)],
                                    'red_floor1': [(0, 43, 46), (10, 255, 255)],
                                    'red_floor2': [(156, 43, 46), (180, 255, 255)],
                                    'green_bridge': [(35, 43, 20), (100, 255, 255)],
                                    'green': [(47, 0, 135), (255, 110, 255)],    # 官方
                                    'yellow_hole': [(10, 70, 46), (34, 255, 255)],
                                    'black_hole': [(0, 0, 0), (180, 255, 80)],
                                    'black_gap': [(0, 0, 0), (180, 255, 100)],
                                    'black_dir': [(0, 0, 0), (180, 255, 46)],
                                    'blue': [(110, 43, 46), (124, 255, 255)],
                                    'black_door': [(0, 0, 0), (180, 255, 46)]}
        self.color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                            'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},   # 比上面那个好用
                            'green': {'Lower': np.array([35, 43, 46]), 'Upper': np.array([77, 255, 255])},
                            'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
                            'black_obscle': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 60])},
                            'black_dir': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
                            'yellow': {'Lower': np.array([10, 43, 46]), 'Upper': np.array([34, 255, 255])},
                           }
        self.start_door_detectline = 40
        self.start_door_threshold = 20
        self.start_door_count = 0
        # self.start_door_init()

    def start_door_init(self, video_img):
        """
        func: 开始关卡初始化，确定闸门识别线
        return: None
        """
        video_img = cv2.copyMakeBorder(video_img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))
        frame_gauss = cv2.GaussianBlur(video_img, (3, 3), 0)
        frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)
        frame_door1 = cv2.inRange(frame_hsv, self.color_dist['yellow']['Lower'],
                                  self.color_dist['yellow']['Upper'])
        frame_door2 = cv2.inRange(frame_hsv, self.color_dist['black']['Lower'],
                                  self.color_dist['black']['Upper'])
        frame_door = cv2.add(frame_door1, frame_door2)
        opened = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # show_img = np.copy(closed)
        # cv2.line(show_img, (int(len(closed[0])/2), 0), (int(len(closed[0])/2), closed.shape[0]), (0, 0, 255),
        #          1)
        # cv2.imshow('test1', show_img)
        # cv2.waitKey(0)

        pixel_list = np.array([])
        for i in range(closed.shape[0]):
            pixel_list = np.append(pixel_list, closed[i][int(len(closed[0])/2)])
        whitepixel_location = np.where(pixel_list==255)
        self.start_door_detectline = int(np.mean(whitepixel_location[0])+10)



    def start_door_detect(self, video_img):
        """
        func：实时判断闸门是否抬起
        :param video_img: 单帧图片
        :return: Ture：抬起 No：未抬起
        """
        video_img = cv2.copyMakeBorder(video_img, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255))
        org_img = cv2.resize(video_img, (640, 480), interpolation=cv2.INTER_CUBIC)
        frame_gauss = cv2.GaussianBlur(org_img, (3, 3), 0)
        frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)
        frame_door1 = cv2.inRange(frame_hsv, self.color_dist['yellow']['Lower'],
                                  self.color_dist['yellow']['Upper'])
        frame_door2 = cv2.inRange(frame_hsv, self.color_dist['black']['Lower'],
                                  self.color_dist['black']['Upper'])
        frame_door = cv2.add(frame_door1, frame_door2)
        opened = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        show_img = np.copy(closed)
        cv2.line(show_img, (0, self.start_door_detectline), (len(closed[0]), self.start_door_detectline),
                 (255, 255, 255), thickness=1)
        cv2.imshow('test', show_img)
        cv2.waitKey(0)

        self.start_door_count = 0
        for i in closed[self.start_door_detectline]:
            if i == 255:
                self.start_door_count  += 1
        if self.start_door_count  >= self.start_door_threshold:
            return False
        else:
            return True

    def hole_path_detect(self, video_img):
        """
        func：实时检测路径
        :param video_img:单帧图片
        :return:机器人视野中路径的偏离程度，
        """
        org_img = cv2.resize(video_img, (640, 480), interpolation=cv2.INTER_CUBIC)
        frame_gauss = cv2.GaussianBlur(org_img, (3, 3), 0)
        frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)
        frame_greenpath = cv2.inRange(frame_hsv,  self.color_dist['green']['Lower'],
                                  self.color_dist['green']['Upper'])
        frame_inv = cv2.bitwise_not(frame_greenpath)
        opened = cv2.morphologyEx(frame_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        img_d = cv2.dilate(closed, kernel)

        show_img = np.copy(org_img)

        contours, hierarchy = cv2.findContours(img_d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_len = cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, 0.02*contour_len, True) #对轮廓进行多边形逼近
            if len(contour) == 4 and cv2.isContourConvex(contour) and cv2.contourArea(contour) > 1000: #判断符合条件的四边形
                contour_2d = [[x[0][0], x[0][1]] for x in contour]
                contour_2d = sorted(contour_2d, key=(lambda x:x[0]), reverse=True)
                if contour_2d[0][1] <= contour_2d[1][1]:
                    temp = contour_2d[0]
                    contour_2d[0] = contour_2d[1]
                    contour_2d[1] = temp
                right_line = [contour_2d[0], contour_2d[1]]
                areaS = cv2.contourArea(contour)
                print(contour_2d)
                print(right_line)
                print(areaS)
                cv2.line(show_img, (contour_2d[0][0], contour_2d[0][1]), (contour_2d[1][0], contour_2d[1][1]),
                         (0, 0, 255), thickness=10)
                cv2.line(show_img, (contour_2d[1][0], contour_2d[1][1]), (contour_2d[2][0], contour_2d[2][1]),
                         (255, 0, 0), thickness=10)
                cv2.line(show_img, (contour_2d[2][0], contour_2d[2][1]), (contour_2d[3][0], contour_2d[3][1]),
                         (0, 255, 0), thickness=10)
                cv2.line(show_img, (contour_2d[3][0], contour_2d[3][1]), (contour_2d[0][0], contour_2d[0][1]),
                         (255, 255, 255), thickness=10)

        cv2.imshow('org', show_img)
        # cv2.waitKey(0)

        # print(math.atan2(right_line[0][1]-right_line[1][1],right_line[1][0]-right_line[0][0])*(180/math.pi))
        # print((right_line[0][0]+right_line[1][0])/2)
        return math.atan2(right_line[0][1]-right_line[1][1], right_line[1][0]-right_line[0][0])*(180/math.pi),\
               (right_line[0][0]+right_line[1][0])/2, areaS




# if __name__ == "__main__":
#     img = cv2.imread('./pic/start_door.PNG')
#     object_detect = OBJECT_DETECT()
#     object_detect.start_door_detect(img)