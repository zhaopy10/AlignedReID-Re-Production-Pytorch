# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import pickle
import numpy as np

# Remember to add your installation path here
# Option a
#dir_path = os.path.dirname(os.path.realpath(__file__))
#if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
#else: sys.path.append('../../python');
sys.path.append('/home/corp.owlii.com/peiyao.zhao/pose_detection/openpose/build/python')

# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

dataset_path = '/home/corp.owlii.com/peiyao.zhao/reid/dataset/douyin_0821/'

if True:
    partition_file = dataset_path + 'partitions.pkl'
    f = open(partition_file, 'rb')
    data = pickle.load(f)
    test_im_names = data['test_im_names']
    print('img number:', len(test_im_names))
    
    for fname in test_im_names:
        fname_full = dataset_path + 'images/' + fname
        print('load', fname_full)
        output_name = fname_full + '.txt'
        output_img_name = fname_full + '.result.jpg'
        img = cv2.imread(fname_full)
        keypoints, output_image = openpose.forward(img, True)
        cv2.imwrite(output_img_name, output_image)
        person_num = keypoints.shape[0]
        if not person_num==1:
            print(person_num, 'detected')
        else:
            np.savetxt(output_name, keypoints[0,:,:])
            
    
    # Read new image
    #img = cv2.imread("../../../examples/media/COCO_val2014_000000000192.jpg")
    
    # Output keypoints and the image with the human skeleton blended on it
    
    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    #print(keypoints.shape)
    #print(keypoints)
    # Display the image
    #cv2.imshow("output", output_image)
    #cv2.imwrite("result.jpg", output_image)
    #cv2.waitKey(15)
