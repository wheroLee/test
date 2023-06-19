### Read video
def createMarker(_data):
    import numpy as np
    import cv2, PIL
    from cv2 import aruco
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # _data = r"D:\python"

    fig = plt.figure()
    nx = 4
    ny = 3
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.drawMarker(aruco_dict,i, 700)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")

    plt.savefig(_data+"\\markers.pdf")
    plt.show()

### Read video
def readVideo(_file):
    import cv2
    cap=cv2.VideoCapture(_file)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Output", gray)
        else:
            break
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

### Detect from video source
def detectMarkerFromVideo(_file):
    import cv2
    from cv2 import aruco
    import matplotlib.pyplot as plt

    dt = 1/24 #sec
    t  = 0.0
    
    fi = open(r"/home/wrl/Documents/python/coord.txt", 'w')
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
       

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1)
        dic = {}
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
            for id in ids:
                # print(id[0])
                
                dic[f"id{id[0]}"]=[time, id[0],frame_markers[0][id[0]][0],frame_markers[0][id[0]][1],frame_markers[0][id[0]][2]]
                # print(dic)
            for i in lstID:
                if f"id{i}" in dic:
                    print(f"id{i} is")
                    print(dic[f"id{ids[i][0]}"])
                    # fi.write( dic[f"id{ids[i][0]}"] + '\n' )
                else:
                    print(f"id{i} is not")
                    fi.write( f"{time}, {i}, null, null, null" +'\n' )     
            # print(f"id{ids[0][0]}")
            #print( str(time)+" : "+str(ids) +" : "+str(corners))
            fi.write( str(time)+" : "+str(ids) +" : "+str(corners)+'\n')
            cv2.imshow("Output",frame_markers)
            
            # 인식된 이미지 파일로 저장
            out.write(frame_markers)

        else:
            break
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    fi.close()
    cap.release()
    #저장 파일 종료
    out.release()
    cv2.destroyAllWindows()

### Detect from realtime camera
def detectMarkerFromCamera():
    import cv2
    from cv2 import aruco
    import matplotlib.pyplot as plt

    cap=cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters =  aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            # plt.figure()
            print(str(ids) +",  "+str(corners))
            cv2.imshow("Output",frame_markers)

        else:
            break
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

### Overlay marker ID and video
def overlay( _video ):
    import cv2
    from cv2 import aruco
    import time

    dict_aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    cap = cv2.VideoCapture( _video )

    try:
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            cv2.imshow('frame', frame_markers)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('frame')
        cap.release()
    except KeyboardInterrupt:
        cv2.destroyWindow('frame')
        cap.release()

# def quad_area(data):
#     import numpy as np
#     import pandas as pd
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
#                    index = pd.MultiIndex.from_product(
#                            [ids.flatten(), ["c{0}".format(i )for i in np.arange(4)+1]],
#                        names = ["marker", ""] ))

# data = data.unstack().swaplevel(0, 1, axis = 1).stack()
# data["m1"] = data[["c1", "c2"]].mean(axis = 1)
# data["m2"] = data[["c2", "c3"]].mean(axis = 1)
# data["m3"] = data[["c3", "c4"]].mean(axis = 1)
# data["m4"] = data[["c4", "c1"]].mean(axis = 1)
# data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis = 1)
# data
