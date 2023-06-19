#-*-coding:utf-8-*-

import sys
print("python version : " + sys.version)

import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# #%%  lib check and install
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
import block
_data = r"/home/wrl/Documents/python"
_file = r'./test000.mp4'

#%% main
if __name__=="__main__":
    # ret=block.createMarker(_data)
    # ret=block.readVideo(_file)
    ret=block.detectMarkerFromVideo(_file)
    # ret=block.detectMarkerFromCamera()

else:
    print("Imported module")
    print(__name__)