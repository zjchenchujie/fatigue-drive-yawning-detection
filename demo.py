import sys
sys.path.append('/home/aia1/ccj/caffe-master/python')
import caffe
import cv2

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import utils.face_detector as face_det
import copy


def drawBoxes(im, boxes, conf):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    # score = boxes[4]

    cv2.putText(im, '%.4f' % conf[0], (int(x1), int(y1)), 5, 1.0,
                [255, 0, 0])
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    return im



test_listfile = 'demo.lst'

# face_model_path = './mtcnn-master/model'
yawn_model_file = './model/deploy.prototxt'
yawn_weights_file = './model/yawning_iter_10000.caffemodel'
save_video_file = 'yawn_detection_demo.avi'
debug = True

# parameters of face detection

minsize = 20
smooth_nframe = 10
net_w = 112
net_h = 112

# parameters of yawn detection
c3d_depth = 16
sampling_rate = 2
num_slide = 4
yawn_label = 2

# parameters of plot
num_time = 20

# Initialize yawn detection network
caffe.set_mode_gpu()
caffe.set_device(0)
yawn_net = caffe.Net(yawn_model_file, yawn_weights_file, caffe.TEST)

fd = open(test_listfile, 'r')
for video_path in fd.readlines():
    fig = plt.figure('Yawn Detection')
    ax1 = fig.add_axes([0, 0, 1.0, 1.0])
    ax2 = fig.add_axes([0.1, 0.65, 0.3, 0.3], facecolor=[0.2,0.2,0.2])

    video_path = video_path.strip('\n')

    cap = cv2.VideoCapture(video_path)
    ftotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    writer = cv2.VideoWriter(save_video_file, fourcc=fourcc, fps=5, frameSize=(fwidth, fheight))

    fig_cnt = 0
    pre_nframe_bboxes = []
    pre_nframe_face = []
    pre_nframe_prob = []
    for fcnt in range(ftotal):
        success, raw_img = cap.read()
        if fcnt % sampling_rate:
            continue

        conf, raw_boxes = face_det.get_facebox(image=raw_img, threshold=0.9)
        # Store history bbox
        if len(raw_boxes) == 0:
            # if face can't be detected between n frames
            if len(pre_nframe_bboxes) == 0:
                print('***too sad, the face can not be detected!')
                # break
            else:
                # print('---missing detect!')
                bbox = np.average(pre_nframe_bboxes, 0)
        elif len(raw_boxes) == 1:
            bbox = raw_boxes[0]
        elif len(raw_boxes) > 1:
            # print('---multi detect!')
            idx = np.argsort(conf)
            bbox = raw_boxes[idx[0]]

        # Crop and scale detected face
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        face = raw_img[y:y + h, x:x + w, :]
        face = cv2.resize(face, (net_w, net_h))
        # cv2.imwrite('face.jpg', face)

        if debug:
            #cv2.imshow('face', face)
            if len(bbox) == 0:
                continue
            img = drawBoxes(raw_img, bbox, conf)
            #cv2.imshow('img', img)
            #ch = cv2.waitKey(1) & 0xFF
            #if ch == 27:
            #    break


            in_blobs = np.zeros([1, 3, net_h, net_w], dtype=np.float64)
            face = np.transpose(face, [2, 0, 1])


            face = face - 127.5
            face = face * 0.0078125

            in_blobs[0, :, :, :] = face

            # Network forward
            yawn_net.blobs['data'].data[...] = in_blobs
            probs = yawn_net.forward()['prob'][0]
            print(probs)

            # Strore prob
            if len(pre_nframe_prob) >= num_time:
                pre_nframe_prob.pop(0)
            pre_nframe_prob.append(probs[1])

            # Draw plot
            show_prob = np.zeros([num_time], dtype=np.float32)
            for i in range(len(pre_nframe_prob)):
                show_prob[i] = pre_nframe_prob[i]
            show_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

            plt.axes(ax1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.imshow(show_img)

            plt.axes(ax2)
            plt.cla()
            plt.xlim((0, num_time))
            plt.ylim((-0.10, 1.1))
            font = {'color': 'red'}
            plt.xlabel('Time', fontdict=font)
            plt.ylabel('Yawn Probability', fontdict=font)
            plt.plot(range(num_time), show_prob, 'c')
            plt.pause(0.001)

            # Save figure to disk
            fig_cnt += 1
            plt.savefig('image.png')

            fig_img = cv2.imread('image.png')
            writer.write(fig_img)

    cap.release()
    writer.release()
