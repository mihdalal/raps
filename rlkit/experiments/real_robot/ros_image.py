#!/usr/bin/env python3
from typing import Optional

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import pyrender

import rospy
from autolab_core.camera_intrinsics import CameraIntrinsics
from autolab_core.image import RgbdImage
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


class RealsenseROSCamera:
    """ """

    def __init__(
        self,
        camera_id=2,
        rgb_topic: Optional[str] = "/camera/color/image_raw",
        intrinsics_topic: Optional[str] = "/camera/color/camera_info",
    ):
        rgb_topic = rgb_topic[0:7] + str(camera_id) + rgb_topic[7:]
        intrinsics_topic = intrinsics_topic[0:7] + str(camera_id) + intrinsics_topic[7:]

        self.rgb_topic = rgb_topic
        self.intrinsics_topic = intrinsics_topic
        self.bridge = CvBridge()

        # Create intrinsics
        camera_info_msg = rospy.wait_for_message(self.intrinsics_topic, CameraInfo)
        self.rs_intrinsics = camera_info_msg_to_rs_intrinsics(camera_info_msg)

        # autolab core intrinsics
        self.intrinsics = CameraIntrinsics(
            frame="camera",
            fx=self.rs_intrinsics.fx,
            fy=self.rs_intrinsics.fy,
            cx=self.rs_intrinsics.ppx,
            cy=self.rs_intrinsics.ppy,
            skew=0.0,
            height=self.rs_intrinsics.height,
            width=self.rs_intrinsics.width,
        )

    def get_image(self):

        rgb = None

        # Get color image
        if self.rgb_topic is not None:
            rgb_msg = rospy.wait_for_message(self.rgb_topic, Image)
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        return rgb

    def make_pyrender_scene(self):
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([1.0, 1.0, 1.0]),
        )

        cam = pyrender.IntrinsicsCamera(
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
        )
        scene.add(cam, pose=np.eye(4))

        return scene


def camera_info_msg_to_rs_intrinsics(camera_info_msg):
    rs_intrinsics = rs.intrinsics()
    rs_intrinsics.width = camera_info_msg.width
    rs_intrinsics.height = camera_info_msg.height
    rs_intrinsics.ppx = camera_info_msg.K[2]
    rs_intrinsics.ppy = camera_info_msg.K[5]
    rs_intrinsics.fx = camera_info_msg.K[0]
    rs_intrinsics.fy = camera_info_msg.K[4]
    # rs_intrinsics.model = camera_info_msg.distortion_model
    rs_intrinsics.model = rs.distortion.brown_conrady  # plumb bob
    rs_intrinsics.coeffs = [i for i in camera_info_msg.D]
    return rs_intrinsics


"""
cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
"""


def main():
    cam = RealsenseROSCamera()
    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX

        img = cam.get_image()[:, :, ::-1]
        lowerBound = np.array([155, 25, 0])
        upperBound = np.array([179, 255, 255])
        img = cv2.resize(img, (340, 220))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        conts, h = cv2.findContours(
            maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        largest_area = 0
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            largest_area = max(largest_area, w * h)
        for i in range(len(conts)):
            x, y, w, h = cv2.boundingRect(conts[i])
            if w * h == largest_area:
                # cv2.drawContours(img,conts[i:i+1],-1,(255,0,0),3)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(
                    img, "Die", (x, y + h), font, 1, (0, 255, 255), 1, cv2.LINE_AA
                )
                break
        cv2.imshow("mask", mask)
        cv2.imshow("cam", img)
        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)

        cv2.waitKey(10)


if __name__ == "__main__":
    rospy.init_node("camera", anonymous=True)
    camera = RealsenseROSCamera()
    main()
