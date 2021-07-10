import object_detect
import cv2



class RUN():
    def __init__(self):
        self.runcode = 2
        self.OBJECT_DETECT = object_detect.OBJECT_DETECT()
        self.video = cv2.VideoCapture(0)

    def pass_hole(self, video_img):
        while self.runcode == 2:
            theta, right_dist, areaS = self.OBJECT_DETECT.hole_path_detect(video_img)
            if areaS >= 20000:
                if right_dist >= 100:
                    print("机器人右平移")
                else:
                    print("机器人调整姿态")
                    if theta >= 95:
                        print("机器人原地左转")
                    elif theta <= 85:
                        print("机器人原地右转")
                    else:
                        print("机器人直行")
            else:
                print("Passed  Hole Successfully")
                self.runcode = 3

    def test_pass_hole(self):
        for i in range(6):
            raw_img = cv2.imread("./pic/2-{}.jpg".format(i+1))
            theta, right_dist, areaS = self.OBJECT_DETECT.hole_path_detect(raw_img)
            print("-------pic{} start--------".format(i+1))
            print("theta", theta)
            print("right_dist", right_dist)
            print("areaS", areaS)
            print("-------pic{} end--------".format(i + 1))
            cv2.waitKey(0)

if __name__ == "__main__":
    main_mission = RUN()
    main_mission.test_pass_hole()