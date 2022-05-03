import argparse
import json
import os

import cv2
from draw_util import draw_2D_box, init_colors


def parse_args():
    """Function for parsing input arguments

    Returns:
        (object): object containing passed arguments
    """
    parser = argparse.ArgumentParser(description='Neural network training',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_file',
                        default = "/home/darknet/result.json", type = str,
                        help = 'specify path to Darknet results json file')
    parser.add_argument('--save_dir',
                        default = "/home/output/real_photos/", type = str,
                        help = 'specify folder for saving images')
    parser.add_argument('--classes_file',
                        default = "/home/darknet/data/vehicle_classes.names", type = str,
                        help = 'specify path to file with class names')
    parser.add_argument('--confidence',
                        default = 0.0, type = float,
                        help = 'set condfidence threshold. Value from range [0, 1].')
    parsed_args = parser.parse_args()
    parsed_args.classes = read_txt_file(parsed_args.classes_file)
    parsed_args.colors = init_colors(len(parsed_args.classes))
    return parsed_args

def read_txt_file(filepath):
    """ Read lines from text file.

    Args:
        filepath (str): Path to file

    Returns:
        (list): lines from file
    """
    return [line.strip() for line in open(filepath).readlines()]

def parse_detections(args):
    """Parse Yolo detections from results file.

    Args:
        args (obj): object with input arguments
    """

    with open(args.json_file) as f:
        images_data = json.load(f)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    for image in images_data:
        img_og = cv2.imread(image["filename"])
        print('Processing image:', image["filename"])
        image_name = image["filename"][image["filename"].rindex(os.path.sep)+1:]
        img_cpy = img_og.copy()
        img_height, img_width, _ = img_og.shape
        object_list = image["objects"]
        for count, obj in enumerate(object_list):
            if obj["confidence"] >= args.confidence:
                bbox_message = "%d. %s (%.2f)" % (count, obj["name"], obj["confidence"]*100)
                center_x = obj["relative_coordinates"]["center_x"] * img_width
                center_y = obj["relative_coordinates"]["center_y"] * img_height
            
                width = obj["relative_coordinates"]["width"] * img_width
                height = obj["relative_coordinates"]["height"] * img_height
                bbox = np.array([(center_x - width / 2), (center_y - height / 2), (center_x + width / 2), (center_y + height / 2)])
                draw_2D_box(img_cpy, bbox.to_numpy_array(), args.colors, args.classes.index(obj["name"]), bbox_message, show_label=True)
        cv2.imwrite(args.save_dir + image_name, img_cpy)

    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    parse_detections(args)
