### Extract coordinates of a marker
import cv2
from cv2 import aruco
import numpy as np
import time


class MarkSearch:
    dict_aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    def __init__(self, cameraID):
        self.cap = cv2.VideoCapture(cameraID)

        _file = r"/home/ubun/Documents/python/test002.mp4"
        self.cap = cv2.VideoCapture(_file)

    def get_mark_coordinate(self, num_id):
        """
        Obtain marker id list from still image
        """
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, dict_aruco, parameters=parameters
        )

        ### If marker of num_id is detected ###
        denu = {}
        for num_id in np.ravel(ids):
            index = np.where(ids == num_id)[0][0]  # Extract index of num_id
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [(cornerUL[0] + cornerBR[0]) / 2, (cornerUL[1] + cornerBR[1]) / 2]
            denu[f"s{num_id}"] = {center[0], center[1]}
            print(denu)
            # print('Upper left : {}'.format(cornerUL))
            # print('Upper right : {}'.format(cornerUR))
            # print('Lower right : {}'.format(cornerBR))
            # print('Lower Left : {}'.format(cornerBL))
            # print('Center : {}'.format(center))

            # print(corners[index])

            return center

        return None


if __name__ == "__main__":
    import cv2
    from cv2 import aruco
    import numpy as np
    import time

    dict_aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    ### --- parameter --- ###
    cameraID = 0
    cam0_mark_search = MarkSearch(cameraID)

    markID = 1

    try:
        print(" ----- get_mark_coordinate ----- ")
        while True:
            # print(' ----- get_mark_coordinate ----- ')
            ret = cam0_mark_search.get_mark_coordinate(markID)
            # time.sleep(0.5)
    except KeyboardInterrupt:
        cam0_mark_search.cap.release()
