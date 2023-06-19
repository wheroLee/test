import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# #%% create marker

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# _data = r"D:\python"

# fig = plt.figure()
# nx = 4
# ny = 3
# for i in range(1, nx*ny+1):
#     ax = fig.add_subplot(ny,nx, i)
#     img = aruco.drawMarker(aruco_dict,i, 700)
#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     ax.axis("off")

# plt.savefig("D:\python/markers.pdf")
# #plt.show()


#%% open image
frame = cv2.imread("/home/ubun/Documents/python/aruco_photo2.jpg")
# plt.figure()
# plt.imshow(frame)
#plt.show()


#%% 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
# print("corners          : " + str(corners))
# print("ids              : " + str(ids))
# print("rejectedImgPoints: " + str(rejectedImgPoints))
# print("frame_markers    : " + str(frame_markers))

### If marker of num_id is detected ###
if num_id in np.ravel(ids) :
    index = np.where(ids == num_id)[0][0] #Extract index of num_id
    cornerUL = corners[index][0][0]
    cornerUR = corners[index][0][1]
    cornerBR = corners[index][0][2]
    cornerBL = corners[index][0][3]

    center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]

    print('Upper left : {}'.format(cornerUL))
    print('Upper right : {}'.format(cornerUR))
    print('Lower right : {}'.format(cornerBR))
    print('Lower Left : {}'.format(cornerBL))
    print('Center : {}'.format(center))

    print(corners[index])

if __name__ == "__main__" :

    import cv2
    from cv2 import aruco
    import numpy as np
    import time

    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    ### --- parameter --- ###
    cameraID = 0
    cam0_mark_search = MarkSearch(cameraID)

    markID = 1

    try:
        while True:
            print(' ----- get_mark_coordinate ----- ')
            print(cam0_mark_search.get_mark_coordinate(markID))
            time.sleep(0.5)
    except KeyboardInterrupt:
        cam0_mark_search.cap.release()
#     return center

# return None

# # %%
# plt.figure()
# plt.imshow(frame_markers)
# for i in range(len(ids)):
#     c = corners[i][0]
#     plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
# plt.legend()
# plt.show()

# # %% 
# import pip
# from imutils.video import VideoStream
# import argparse
# import imutils
# import time
# import cv2
# import sys

# # %%
# ap=argparse.ArgumentParser()
# ap.add_argument("-t","--type", type=str,default="DICT_ARUCO_ORIGINAL",help="type of ArUCo tag to detect")
# args=vars(ap.parse_args())


# %% read video
# import cv2
# cap=cv2.VideoCapture(r'D:\python\video.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# %% read video 2
# import cv2
# cap=cv2.VideoCapture(r'D:\python\video.mp4')

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow("Output", gray)
#     else:
#         break
#     key = cv2.waitKey(1) & 0xFF
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()

# # %% save video
# import cv2

# cap = cv2.VideoCapture(0)
# # cap=cv2.VideoCapture(r'D:\python\video.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter(r'D:\python\output.mp4', fourcc, 25.0, (640,480))

# while (cap.isOpened()):
#     ret, frame = cap.read()

#     if ret:
#         # 이미지 반전,  0:상하, 1 : 좌우
#         frame = cv2.flip(frame, 0)

#         out.write(frame)

#         cv2.imshow('frame', frame)

#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()
