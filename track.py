import sys
print("python version : " + sys.version)

import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# %matplotlib nbagg

#%% Create Marker
def createMarker():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    fig = plt.figure()
    nx = 4
    ny = 3
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.drawMarker(aruco_dict,i, 700)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")

    plt.savefig("./markers.pdf")
    plt.show()

def markerTracking(_file):
    cap=cv2.VideoCapture(_file)
    
    if cap.isOpened() == False:
        print ('Can\'t open the video (%d)' % (_file))
        exit()
        
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #재생할 파일의 높이 얻기
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #재생할 파일의 프레임 레이트 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #저장할 파일 이름
    filename = 'marker_detect.avi'     
    
    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1)    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        time = cap.get(cv2.CAP_PROP_POS_MSEC)       
        
        cv2.imshow("Output",frame_markers)
            
        # 인식된 이미지 파일로 저장
        out.write(frame_markers)

        # plt.figure()
        # plt.imshow(frame_markers)
        # for i in range(len(ids)):
        #     c = corners[i][0]
        #     plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
        # plt.legend()
        # plt.show()

        # def quad_area(data):
        #     l = data.shape[0]//2
        #     corners = data[["c1", "c2", "c3", "c4"]].values.reshape(l, 2,4)
        #     c1 = corners[:, :, 0]
        #     c2 = corners[:, :, 1]
        #     c3 = corners[:, :, 2]
        #     c4 = corners[:, :, 3]
        #     e1 = c2-c1
        #     e2 = c3-c2
        #     e3 = c4-c3
        #     e4 = c1-c4
        #     a = -.5 * (np.cross(-e1, e2, axis = 1) + np.cross(-e3, e4, axis = 1))
        #     return a

        # corners2 = np.array([c[0] for c in corners])

        # data = pd.DataFrame({"x": corners2[:,:,0].flatten(), "y": corners2[:,:,1].flatten()},
        #                 index = pd.MultiIndex.from_product(
        #                         [ids.flatten(), ["c{0}".format(i )for i in np.arange(4)+1]],
        #                     names = ["marker", ""] ))

        # data = data.unstack().swaplevel(0, 1, axis = 1).stack()
        # data["m1"] = data[["c1", "c2"]].mean(axis = 1)
        # data["m2"] = data[["c2", "c3"]].mean(axis = 1)
        # data["m3"] = data[["c3", "c4"]].mean(axis = 1)
        # data["m4"] = data[["c4", "c1"]].mean(axis = 1)
        # data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis = 1)
        # data.to_csv('./data.csv', mode='w')
        # print(data)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

_file = r'./markertest.mp4'
#%% main
if __name__=="__main__":
    # ret=block.createMarker(_data)
    # ret=block.readVideo(_file)
    # ret=block.detectMarkerFromVideo(_file)
    ret=markerTracking(_file)
    # ret=block.detectMarkerFromCamera()

else:
    print("Imported module")
    print(__name__)
    
# _file = r'./000.mp4'
# cap=cv2.VideoCapture(_file)

# matrix_coefficients = [fx 0 cx; 0 fy cy; 0 0 1]
# distortion_coefficients=

# def track(matrix_coefficients, distortion_coefficients):
#     while True:
#         ret, frame = cap.read()
#         # operations on the frame come here
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
#         aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Use 5x5 dictionary to find markers
#         parameters = aruco.DetectorParameters_create()  # Marker detection parameters
#         # lists of ids and the corners beloning to each id
#         corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
#         if np.all(ids is not None):  # If there are markers found by detector
#             for i in range(0, len(ids)):  # Iterate in markers
#                 # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
#                 rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                            distortion_coefficients)
#                 (rvec - tvec).any()  # get rid of that nasty numpy value array error
#                 aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
#                 aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
#         # Display the resulting frame
#         cv2.imshow('frame', frame)
#         # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
#         key = cv2.waitKey(3) & 0xFF
#         if key == ord('q'):  # Quit
#             break
    
#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()
    
# track()