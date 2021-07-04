import object_detect
import cv2



class RUN():
    def __init__(self):
        self.runcode = 2
        self.OBJECT_DETECT = object_detect.OBJECT_DETECT('./pic/round2-1.jpg')
        self.video = cv2.VideoCapture(0)

    def pass_hole(self):
        while self.runcode == 2:
            ret, frame = self.video.read()
            print(frame.shape)
            theta, right_dist, areaS = self.OBJECT_DETECT.hole_path_detect(frame)
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

if __name__ == "__main__":
    main_mission = RUN()
    main_mission.pass_hole()