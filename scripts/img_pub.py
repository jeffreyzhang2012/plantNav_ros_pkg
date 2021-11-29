#!/usr/bin/env python3

import cv2 as cv
import threading
import random
import time
from time import sleep
# import ipywidgets as widgets
# from IPython.display import display
# from color_follow import color_follow
# import Arm_Lib
import rospy
from sensor_msgs.msg import JointState, Image
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud
import sensor_msgs.msg
import sys
# from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform 
from scipy.spatial import KDTree

class CvBridgeError(TypeError):
    """
    This is the error raised by :class:`cv_bridge.CvBridge` methods when they fail.
    """
    pass


class CvBridge(object):
    """
    The CvBridge is an object that converts between OpenCV Images and ROS Image messages.
       .. doctest::
           :options: -ELLIPSIS, +NORMALIZE_WHITESPACE
           >>> import cv
           >>> import numpy as np
           >>> from cv_bridge import CvBridge
           >>> br = CvBridge()
           >>> dtype, n_channels = br.encoding_as_cvtype2('8UC3')
           >>> im = np.ndarray(shape=(480, 640, n_channels), dtype=dtype)
           >>> msg = br.cv_to_imgmsg(im)  # Convert the image to a message
           >>> im2 = br.imgmsg_to_cv(msg) # Convert the message to a new image
           >>> cmprsmsg = br.cv_to_compressed_imgmsg(im)  # Convert the image to a compress message
           >>> im22 = br.compressed_imgmsg_to_cv(msg) # Convert the compress message to a new image
           >>> cv.imwrite("this_was_a_message_briefly.png", im2)
    """

    def __init__(self):
        self.cvtype_to_name = {}
        self.cvdepth_to_numpy_depth = {cv.CV_8U: 'uint8', cv.CV_8S: 'int8', cv.CV_16U: 'uint16',
                                       cv.CV_16S: 'int16', cv.CV_32S:'int32', cv.CV_32F:'float32',
                                       cv.CV_64F: 'float64'}

        for t in ["8U", "8S", "16U", "16S", "32S", "32F", "64F"]:
            for c in [1, 2, 3, 4]:
                nm = "%sC%d" % (t, c)
                self.cvtype_to_name[getattr(cv, "CV_%s" % nm)] = nm

        self.numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                        'int16': '16S', 'int32': '32S', 'float32': '32F',
                                        'float64': '64F'}
        self.numpy_type_to_cvtype.update(dict((v, k) for (k, v) in self.numpy_type_to_cvtype.items()))

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        return '%sC%d' % (self.numpy_type_to_cvtype[dtype.name], n_channels)

    def cvtype2_to_dtype_with_channels(self, cvtype):
        from cv_bridge.boost.cv_bridge_boost import CV_MAT_CNWrap, CV_MAT_DEPTHWrap
        return self.cvdepth_to_numpy_depth[CV_MAT_DEPTHWrap(cvtype)], CV_MAT_CNWrap(cvtype)

    def encoding_to_cvtype2(self, encoding):
        from cv_bridge.boost.cv_bridge_boost import getCvType

        try:
            return getCvType(encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

    def encoding_to_dtype_with_channels(self, encoding):
        return self.cvtype2_to_dtype_with_channels(self.encoding_to_cvtype2(encoding))

    def compressed_imgmsg_to_cv(self, cmprs_img_msg, desired_encoding = "passthrough"):
        """
        Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.
        :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
        :param desired_encoding:  The encoding of the image data, one of the following strings:
           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h
        :rtype: :cpp:type:`cv::Mat`
        :raises CvBridgeError: when conversion is not possible.
        If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
        Otherwise desired_encoding must be one of the standard image encodings
        This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        If the image only has one channel, the shape has size 2 (width and height)
        """
        str_msg = cmprs_img_msg.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                          dtype=np.uint8, buffer=cmprs_img_msg.data)
        im = cv.imdecode(buf, cv.IMREAD_UNCHANGED)

        if desired_encoding == "passthrough":
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, "bgr8", desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def imgmsg_to_cv(self, img_msg, desired_encoding = "passthrough"):
        """
        Convert a sensor_msgs::Image message to an OpenCV :cpp:type:`cv::Mat`.
        :param img_msg:   A :cpp:type:`sensor_msgs::Image` message
        :param desired_encoding:  The encoding of the image data, one of the following strings:
           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h
        :rtype: :cpp:type:`cv::Mat`
        :raises CvBridgeError: when conversion is not possible.
        If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
        Otherwise desired_encoding must be one of the standard image encodings
        This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        If the image only has one channel, the shape has size 2 (width and height)
        """
        dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width),
                           dtype=dtype, buffer=img_msg.data)
        else:
            if(type(img_msg.data) == str):
                im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                         dtype=dtype, buffer=img_msg.data.encode())
            else:
                im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                               dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()

        if desired_encoding == "passthrough":
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def cv_to_compressed_imgmsg(self, cvim, dst_format = "jpg"):
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::CompressedImage message.
        :param cvim:      An OpenCV :cpp:type:`cv::Mat`
        :param dst_format:  The format of the image data, one of the following strings:
           * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
           * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#Mat imread(const string& filename, int flags)
           * bmp, dib
           * jpeg, jpg, jpe
           * jp2
           * png
           * pbm, pgm, ppm
           * sr, ras
           * tiff, tif
        :rtype:           A sensor_msgs.msg.CompressedImage message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``format``
        This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        """
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError('Your input type is not a numpy array')
        cmprs_img_msg = sensor_msgs.msg.CompressedImage()
        cmprs_img_msg.format = dst_format
        ext_format = '.' + dst_format
        try:
            cmprs_img_msg.data = np.array(cv.imencode(ext_format, cvim)[1]).tostring()
        except RuntimeError as e:
            raise CvBridgeError(e)

        return cmprs_img_msg

    def cv_to_imgmsg(self, cvim, encoding = "passthrough"):
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.
        :param cvim:      An OpenCV :cpp:type:`cv::Mat`
        :param encoding:  The encoding of the image data, one of the following strings:
           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h
        :rtype:           A sensor_msgs.msg.Image message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``
        If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
        Otherwise desired_encoding must be one of the standard image encodings
        This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        """
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError('Your input type is not a numpy array')
        img_msg = sensor_msgs.msg.Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            # Verify that the supplied encoding is compatible with the type of the OpenCV image
            if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
                raise CvBridgeError("encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type))
        if cvim.dtype.byteorder == '>':
            img_msg.is_bigendian = True
        img_msg.data = cvim.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg

class particle_filter(object):
    def __init__(self):
        self.n = 400
        self.select = 200
        self.keypoints = np.zeros((self.n,3))
        self.collected_n = 0


    def process(self,kp):
        near_dist = 100
        self.keypoints[:,2] *= 0.7
        # print(len(kp))
        points = np.zeros((len(kp),3))
        for i in range(len(kp)):
            points[i] = np.array((kp[i].pt[0],kp[i].pt[1],kp[i].response))
        dist,idx = KDTree(self.keypoints[:,:2]).query(points[:,:2])
        updatedPoints = points[np.where(dist<near_dist)]
        if(updatedPoints.shape[0] != 0):
            updatedIdx = idx[dist<near_dist]
            updateWeights = updatedPoints[:,2]/self.keypoints[updatedIdx,2]
            self.keypoints[updatedIdx,:2] += updateWeights[:,None] * (updatedPoints[:,:2]-self.keypoints[updatedIdx,:2])
            self.keypoints[updatedIdx,2] += updatedPoints[:,2]
        newPoints = points[np.where(dist>=near_dist)]
        new_and_old_points = np.zeros((self.collected_n+newPoints.shape[0],3))
        new_and_old_points[:self.collected_n] = self.keypoints[:self.collected_n]
        new_and_old_points[self.collected_n:] = newPoints
        new_and_old_points = new_and_old_points[(-new_and_old_points[:,2]).argsort()]
        self.collected_n = min(self.collected_n + newPoints.shape[0], self.n)
        self.keypoints = new_and_old_points[:self.collected_n]
        display_kp = []
        self.keypoints[:,2] = self.keypoints[:,2] * 100 / np.mean(self.keypoints[:,2])
        for i in range(min(self.collected_n,self.select)):
            # print(self.keypoints[i,2])
            if self.keypoints[i,2] > 50:
                display_kp.append(cv.KeyPoint(self.keypoints[i,0],self.keypoints[i,1],self.keypoints[i,2]))
        # print("HI")
        return display_kp
            


class Detector(object):
    def __init__(self):
        capture = cv.VideoCapture(0)
        capture.set(3, 640)
        capture.set(4, 480)
        capture.set(5, 30)  #set frame
        self.capture = capture
        self.fast = cv.FastFeatureDetector_create()
        self.fast_thres = 30
        self.fast.setThreshold(self.fast_thres)
        self.target_n = 300
        self.GREEN_MIN = np.array([30, 0, 0],np.uint8)
        self.GREEN_MAX = np.array([100, 255, 255],np.uint8)
        self.pf = particle_filter()
        self.bridge = CvBridge()
        self.prev_img = None
        self.flow = None
        self.hsv = np.zeros((480,640,3),np.uint8)
        self.hsv[...,1] = 255
        # fig = plt.figure()
        # self.ax = fig.addsubplot(projection = 'polar')
    
    def get_img(self):
        st = time.process_time()
        _, img = self.capture.read()
        self.original = cv.resize(img, (640, 480))
        self.img = cv.blur(self.original,(3,3))
        self.gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        print("Acquiring took ", time.process_time()-st)

    def get_masked(self):
        st = time.process_time()
        hsv = cv.cvtColor(self.img,cv.COLOR_BGR2HSV)
        green = cv.inRange(hsv, self.GREEN_MIN, self.GREEN_MAX)
        self.masked = cv.bitwise_and(self.img,self.img,mask=green)
        self.masked_gray = cv.cvtColor(self.masked,cv.COLOR_BGR2GRAY)
        print("Masking took ", time.process_time()-st)


    def fast_detector(self):
        kp = self.fast.detect(self.masked,None)
        diff = len(kp) - self.target_n
        if abs(diff) > 10:
            self.fast_thres += np.sign(diff) * 10
            self.fast.setThreshold(self.fast_thres)
        kp_filtered = self.pf.process(kp)

    def get_opticalFlow(self):
        st = time.process_time()
        if self.prev_img is None: self.prev_img = self.gray
        flow = cv.calcOpticalFlowFarneback(self.prev_img,self.gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.mag, self.ang = cv.cartToPolar(flow[...,0], flow[...,1])
        self.mag = cv.medianBlur(self.mag,3)
        self.ang = cv.medianBlur(self.ang,3)
        self.hsv[...,0] = self.ang*180/np.pi/2
        self.hsv[...,2] = cv.normalize(self.mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(self.hsv,cv.COLOR_HSV2BGR)
        self.prev_img = self.gray
        self.flow = bgr
        print("Optical Flow took ", time.process_time()-st)
        
    def analyzeFlow(self):
        st = time.process_time()
        # idx1d = self.mag.flatten().argsort()[-1000:]
        mag1d = self.mag.flatten()
        ang1d = self.ang.flatten()
        idx1d = np.where(mag1d > 1)
        np.random.shuffle(idx1d[0])
        idx1d = idx1d[0][:2000]
        idx = np.unravel_index((idx1d,), self.mag.shape)
        print(idx,idx1d)
        x = self.mag[idx] * np.cos(self.ang[idx])
        y = self.mag[idx] * np.sin(self.ang[idx])
        print("Sorting took ", time.process_time()-st)
        plt.clf()
        # scatter plot
        plt.scatter(x,y,s=1)
        wid = 50
        plt.xlim([-wid,wid])
        plt.ylim([-wid,wid])
        #  histogram plot
        # n,bins,patches = plt.hist(ang1d[idx1d],50,density=True)
        # plt.xlim([0,2*np.pi])

        plt.pause(0.01)
        print("Analyze Flow took ", time.process_time()-st)


    def to_msg(self,img):
        return self.bridge.cv_to_imgmsg(np.array(img))

    
        

# Arm.Arm_serial_servo_write6_array(joints, 100)
def publish():
    pub_raw = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    pub_masked = rospy.Publisher("/camera/image_masked", Image, queue_size=10)
    pub_flow = rospy.Publisher("/camera/image_opticalFlow", Image, queue_size=10)
    # pub2 = rospy.Publisher("/camera/image_fast_raw", Image, queue_size=10)
    # pub3 = rospy.Publisher("/camera/image_fast_filtered", Image, queue_size=10)
    # pub4 = rospy.Publisher("/camera/image_edges", Image, queue_size=10)
    d = Detector()
    rospy.init_node("img_publisher", anonymous=True)
    rate = rospy.Rate(1)
    while d.capture.isOpened() and not rospy.is_shutdown():
        try:
            d.get_img()
            d.get_masked()
            d.get_opticalFlow()
            d.analyzeFlow()
            st = time.process_time()
            pub_raw.publish(d.to_msg(d.img))
            pub_masked.publish(d.to_msg(d.masked))
            pub_flow.publish(d.to_msg(d.flow))
            print("Publishing took ", time.process_time()-st)
            # img_fast_raw = cv.drawKeypoints(img,kp,None, color=(255,0,0))
            # img_fast_filtered = cv.drawKeypoints(img,kp_filtered,None, color=(255,0,0))
            # msg = bridge.cv_to_imgmsg(np.array(img))
            # msg_fast_raw = bridge.cv_to_imgmsg(np.array(img_fast_raw))
            # msg_fast_filtered = bridge.cv_to_imgmsg(np.array(img_fast_filtered))
            # pub.publish(msg)
            # pub2.publish(msg_fast_raw)
            # pub3.publish(msg_fast_filtered)
        except KeyboardInterrupt:
            d.capture.release()
            cv.destroyAllWindows()
            break

if __name__ == '__main__':
    try:
        publish()
    except rospy.ROSInterruptException:
        pass
