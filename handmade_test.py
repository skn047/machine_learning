import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import xml.etree.ElementTree as ET


# 検出箇所を矩形で描画
def draw_rectangles(img, pred_rects, ground_truths):
    #  left, right, top, bottom
    cv2.line(img, (int(pred_rects[0]), int(pred_rects[2])), (int(pred_rects[0]), int(pred_rects[3])), (255, 0 , 255), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(pred_rects[0]), int(pred_rects[2])), (int(pred_rects[1]), int(pred_rects[2])), (255, 0 , 255), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(pred_rects[0]), int(pred_rects[3])), (int(pred_rects[1]), int(pred_rects[3])), (255, 0 , 255), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(pred_rects[1]), int(pred_rects[2])), (int(pred_rects[1]), int(pred_rects[3])), (255, 0 , 255), thickness=6, lineType=cv2.LINE_AA)
    # print("letf:{}".format(int(pred_rects[0])))
    # print("right:{}".format(int(pred_rects[1])))
    # print("top:{}".format(int(pred_rects[2])))
    # print("bottom:{}".format(int(pred_rects[3])))

    #  top, left, height, width
    cv2.line(img, (int(ground_truths[1]), int(ground_truths[0])), (int(ground_truths[1]), int(ground_truths[0]) + int(ground_truths[2])), (255, 0 , 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(ground_truths[1]), int(ground_truths[0])), (int(ground_truths[1]) + int(ground_truths[3]), int(ground_truths[0])), (255, 0 , 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(ground_truths[1]), int(ground_truths[0]) + int(ground_truths[2])), (int(ground_truths[1]) + int(ground_truths[3]), int(ground_truths[0]) + int(ground_truths[2])), (255, 0 , 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (int(ground_truths[1]) + int(ground_truths[3]), int(ground_truths[0])), (int(ground_truths[1]) + int(ground_truths[3]), int(ground_truths[0]) + int(ground_truths[2])), (255, 0 , 0), thickness=6, lineType=cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def extract_test_rects():
    tree = ET.parse("/home/naoki/dlib-19.1/tools/imglab/build/main/mouth_detector3_test.xml")  #  must change everytime
    root = tree.getroot()

    image_name_coordinates = np.array([]) #  stack dict whose key : filename, value : box_coordinate

    for data in root:
        # print(data.tag)
        for datum in data:
            dic = {}  #  dict whose key : filename, value : box_coordinate
            # print(datum.tag, datum.get(key="file"))
            for single_item in datum:
                # print(single_item.tag, single_item.get(key="top") )

                coordinates = np.array([])
                coordinates = np.append(coordinates, single_item.get(key="top"))
                coordinates = np.append(coordinates, single_item.get(key="left"))
                coordinates = np.append(coordinates, single_item.get(key="height"))
                coordinates = np.append(coordinates, single_item.get(key="width"))
                dic[datum.get(key="file")] = coordinates
                image_name_coordinates = np.append(image_name_coordinates, dic)

    return image_name_coordinates

def evaluation_on_tests(pred_rect, ground_truth):  #  pred_rect : np.array(left, right, top, bottom), ground_truth(top, left, height, width)
    left = max(pred_rect[0], np.float64(ground_truth[1]))
    right = min(pred_rect[1], np.float64(ground_truth[1]) + np.float64(ground_truth[3]))
    top = max(pred_rect[2], np.float64(ground_truth[0]))
    bottom = min(pred_rect[3], np.float64(ground_truth[0]) + np.float64(ground_truth[2]))

    intersection = (right - left) * (bottom - top)
    union = (pred_rect[1] - pred_rect[0]) * (pred_rect[3] - pred_rect[2]) + np.float64(ground_truth[2]) * np.float64(ground_truth[3])

    return intersection / (union - intersection)  #  IoU

detector_path = sys.argv[1]
test_folder_path = sys.argv[2]

detector = dlib.simple_object_detector(detector_path)  #  load detector

ground_truth_rect = extract_test_rects()  #  load ground-truth

count = 0  #  count how many images are included
min_eval = 1.0  #  minimum iou
sum_eval = 0.0
ave_eval = 0.0  #  average iou

for file in glob.glob(os.path.join(test_folder_path, "*.jpg")):
    img = cv2.imread(file)
    rectangles = detector(img)
    rect_coordinates = np.array([])  #  left, right, top, bottom


    for rect in rectangles:
        rect_coordinates = np.append(rect_coordinates, rect.left())
        rect_coordinates = np.append(rect_coordinates, rect.right())
        rect_coordinates = np.append(rect_coordinates, rect.top())
        rect_coordinates = np.append(rect_coordinates, rect.bottom())

    for i in range(len(ground_truth_rect)):
        for name, value in ground_truth_rect[i].items():
            if file == name:
                count += 1
                eval = evaluation_on_tests(rect_coordinates, value)
                sum_eval += eval
                ave_eval = sum_eval / count
                draw_rectangles(img, rect_coordinates, value)
                print("iou : {}".format(round(eval, 3)))
                print("ave_iou : {}".format(round(ave_eval, 3)))
                if min_eval > eval:
                    min_eval = eval
                print("min_iou : {}".format(round(min_eval, 3)))
                print(count)
                print("")
