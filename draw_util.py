import colorsys
import random

import cv2
import numpy as np


def draw_3D_box(image, corners):
    """ Drawing a 3D bounding box around the 3D detected object based on 8 corners that 
        defines it position in the 3D world

    Args:
        image (array): image to draw on 3D bounding box
        points (array): 8 corners of the detected object
    """

    # converting corners into numpy elements
    corners = corners.astype(np.int)
    # drawing 4 lines that defines the depth of the detected object using green color
    for i in range(4):
        first_corner = corners[2 * i]
        second_corner = corners[2 * i + 1]
        cv2.line(image, (first_corner[0], first_corner[1]), (second_corner[0], second_corner[1]), 
                 (0, 255, 0), 2)
    
    # drawing 8 lines that defines the width and length of the detected object using green color
    for i in range(8):
        first_corner = corners[i]
        second_corner = corners[(i + 2) % 8]
        cv2.line(image, (first_corner[0], first_corner[1]), (second_corner[0], second_corner[1]),
                 (0, 255, 0), 2)

    # drawing X sign to define the front side of the detected object using red color
    cv2.line(image, tuple(corners[0]), tuple(corners[7]), (0, 0, 255), 2)
    cv2.line(image, tuple(corners[1]), tuple(corners[6]), (0, 0, 255), 2)

    # drawing X sign to define the upper side of the detected object using purple color
    cv2.line(image, tuple(corners[3]), tuple(corners[7]), (255, 0, 255), 2)
    cv2.line(image, tuple(corners[5]), tuple(corners[1]), (255, 0, 255), 2)

def draw_yolo_bbox(image, bboxes, classes, show_label=True):
    """Draw bounding box for Yolo inference.

    Args:
        image (obj): image to draw on
        bboxes (obj): array of bounding boxes to draw
        classes (dict): classes dict
        show_label (bool, optional): Show class labels. Defaults to True.

    Returns:
        (obj): processed image
    """
    
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    colors = init_colors(num_classes)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def init_colors(classes_num=15):
    hsv_tuples = [(1.0 * x / classes_num, 1., 1.) for x in range(classes_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    print(colors)
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    return colors

def draw_2D_box(image, rectangle_points, colors=None, class_index=None, bbox_message=None, show_label=False, font_scale=0.5):
    """ Drawing a 2D bounding box around the 3D detected object based on 4 corners that defines it 
        position in the 3D world

    Args:
        image (array): image to draw on 3D bounding box
        rectangle_points (array): 4 points of rectangle of the detected object
        index (int, optional): index of the detected vehicle on the image. Defaults to None.
        bbox_message (str, optional): label name of the detected object. Defaults to None.
    """

    rectangle_points = np.ceil(rectangle_points).astype(np.int)
    image_height, image_width, _ = image.shape
    # create tuples for vehicle bounding box
    vehicle_upper_left, vehicle_bottom_right = (rectangle_points[0], rectangle_points[1]), (rectangle_points[2], rectangle_points[3])
    if colors is None:
        bbox_color = (255, 0, 0)
    else:
        bbox_color = colors[class_index]
    bbox_thick = int(0.6 * (image_height + image_width) / 600)
    # drawing rectangle on the image based on the rectangle points
    cv2.rectangle(image, vehicle_upper_left, vehicle_bottom_right, bbox_color, bbox_thick)
    
    if show_label:
            t_size = cv2.getTextSize(bbox_message, 0, font_scale, thickness=bbox_thick // 2)[0]
            text_bottom_right = (vehicle_upper_left[0] + t_size[0], vehicle_upper_left[1] - t_size[1] - 3)
            cv2.rectangle(image, vehicle_upper_left, text_bottom_right, bbox_color, -1) #filled

            cv2.putText(image, bbox_message, (vehicle_upper_left[0], vehicle_upper_left[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
