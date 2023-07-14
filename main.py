#-*-coding:utf-8-*-

import sys
print("python version : " + sys.version)

import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import pandas as pd
#%% lib check and install
# import pipInstall as pip
# liblst = [
#     "opencv-python",
#     "opencv-contrib-python",
#     "opencv-contrib-python==4.6.0.66", 
#     "numpy", "matplotlib", 
#     "pandas"
#     ]
# for lib in liblst:
#     ret = pip.install(lib)

#%% import block
# import block
_data = r"./"
_file = r'F:\_PROJECT_\CON_2023_VSS_AutoLevelingValve\DATAFromIWON\Test_2nd\MOVIE\000.mp4'
_file = r'./markertest.mp4'

### Detect from video source
def detectMarkerFromVideo(_file):
    import cv2
    from cv2 import aruco
    import matplotlib.pyplot as plt
   
    fi = open(r"./coord.csv", 'w')
    cap=cv2.VideoCapture(_file)

    #잘 열렸는지 확인
    if cap.isOpened() == False:
        print ('Can\'t open the video (%d)' % (_file))
        exit()

    #재생할 파일의 넓이 얻기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #재생할 파일의 높이 얻기
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #재생할 파일의 프레임 레이트 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)

    # print('width {0}, height {1}, fps {2}'.format(width, height, fps))

    #XVID가 제일 낫다고 함.
    #linux 계열 DIVX, XVID, MJPG, X264, WMV1, WMV2.
    #windows 계열 DIVX
    #저장할 비디오 코덱
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #저장할 파일 이름
    filename = 'sprite_with_face_detect.avi'

    #파일 stream 생성
    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
    #filename : 파일 이름
    #fourcc : 코덱
    #fps : 초당 프레임 수
    #width : 넓이
    #height : 높이
       
    dic = {}
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1)
        # dic = {}
        lstID = [3,1,2,7,11,12]
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters =  aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            # print(frame_markers[0][0][0])
            # plt.figure()
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            # print(time)
            
            for id in lstID:
                # if id in ids: 
                if id == 1:  
                    print(corners)                               
                    dic[f"id{id}"]=[time, id,frame_markers[0][id][0],frame_markers[0][id][1],frame_markers[0][id][2]]
                    fi.write(f"{time}, {id},{frame_markers[0][id][0]},{frame_markers[0][id][1]},{frame_markers[0][id][2]}"+"\n")
                    # print(f"{time}, {id},{frame_markers[0][id][0]},{frame_markers[0][id][1]},{frame_markers[0][id][2]}"+"\n")
                # if f"id{id[0]}" in dic:
                #     # print(f"id{id[0]}")
                # else:
                #     pass
            # for key in dic:
            #     print key
            #     if f"id{i}" in dic:
            #         # print(f"id{i} is")
            #         print(dic[f"id{ids[i][0]}"])
            #         # fi.write( dic[f"id{ids[i][0]}"] + '\n' )
            #     else:
            #         print(f"id{i} is not")
            #         fi.write( f"{time}, {i}, null, null, null" +'\n' )     
            # # print(f"id{ids[0][0]}")
            # #print( str(time)+" : "+str(ids) +" : "+str(corners))
                # fi.write(dic[f"id{id[0]}"])
                # fi.write( str(time)+" : "+str(ids) +" : "+str(corners)+'\n' )
            
            cv2.imshow("Output",frame_markers)
            
            # 인식된 이미지 파일로 저장
            out.write(frame_markers)

        else:
            break
        # print(dic)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    fi.close()
    cap.release()
    #저장 파일 종료
    out.release()
    cv2.destroyAllWindows()

def quad_area(data):
    l = data.shape[0]//2
    corners = data[["c1", "c2", "c3", "c4"]].values.reshape(l, 2,4)
    c1 = corners[:, :, 0]
    c2 = corners[:, :, 1]
    c3 = corners[:, :, 2]
    c4 = corners[:, :, 3]
    e1 = c2-c1
    e2 = c3-c2
    e3 = c4-c3
    e4 = c1-c4
    a = -.5 * (np.cross(-e1, e2, axis = 1) + np.cross(-e3, e4, axis = 1))
    return a

corners2 = np.array([c[0] for c in corners])
data = pd.DataFrame({"x": corners2[:,:,0].flatten(), "y": corners2[:,:,1].flatten()},
                   index = pd.MultiIndex.from_product(
                           [ids.flatten(), ["c{0}".format(i )for i in np.arange(4)+1]],
                       names = ["marker", ""] ))
data = data.unstack().swaplevel(0, 1, axis = 1).stack()
data["m1"] = data[["c1", "c2"]].mean(axis = 1)
data["m2"] = data[["c2", "c3"]].mean(axis = 1)
data["m3"] = data[["c3", "c4"]].mean(axis = 1)
data["m4"] = data[["c4", "c1"]].mean(axis = 1)
data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis = 1)
data
#%% main
if __name__=="__main__":
    # ret=block.createMarker(_data)
    # ret=block.readVideo(_file)
    # ret=block.detectMarkerFromVideo(_file)
    ret=detectMarkerFromVideo(_file)
    # ret=block.detectMarkerFromCamera()

else:
    print("Imported module")
    print(__name__)