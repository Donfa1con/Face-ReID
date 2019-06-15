import logging
import os
import sys

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


def create_logger():
    lg = logging.getLogger('person_detection')
    lg.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    lg.addHandler(handler)
    return lg


def load_models():
    face_model_xml = os.getenv("FACE_MODEL_PATH") + ".xml"
    face_model_bin = os.getenv("FACE_MODEL_PATH") + ".bin"

    reid_model_xml = os.getenv("REID_MODEL_PATH") + ".xml"
    reid_model_bin = os.getenv("REID_MODEL_PATH") + ".bin"

    mark_model_xml = os.getenv("LANDMARKS_MODEL_PATH") + ".xml"
    mark_model_bin = os.getenv("LANDMARKS_MODEL_PATH") + ".bin"

    plugin = IEPlugin(device="CPU")
    plugin.add_cpu_extension(os.getenv("PLUGIN_PATH"))

    face_net = IENetwork(model=face_model_xml, weights=face_model_bin)
    reid_net = IENetwork(model=reid_model_xml, weights=reid_model_bin)
    mark_net = IENetwork(model=mark_model_xml, weights=mark_model_bin)

    exec_face_net = plugin.load(network=face_net)
    exec_reid_net = plugin.load(network=reid_net)
    exec_mark_net = plugin.load(network=mark_net)

    return exec_face_net, exec_reid_net, exec_mark_net, face_net, reid_net, mark_net


def init_detection_config():
    exec_face_net, exec_reid_net, exec_mark_net, face_net, reid_net, mark_net = load_models()

    _, _, face_net_h, face_net_w = face_net.inputs[next(iter(face_net.inputs))].shape
    _, _, reid_net_h, reid_net_w = reid_net.inputs[next(iter(reid_net.inputs))].shape
    _, _, mark_net_h, mark_net_w = mark_net.inputs[next(iter(mark_net.inputs))].shape

    params = {'face': {'net': exec_face_net,
                       'input_size': {'h': face_net_h,
                                      'w': face_net_w}
                       },

              'reid': {'net': exec_reid_net,
                       'input_size': {'h': reid_net_h,
                                      'w': reid_net_w}
                       },
              'landmarks': {'net': exec_mark_net,
                            'input_size': {'h': mark_net_h,
                                           'w': mark_net_w}
                            }
              }
    return params


def get_face_rect(bbox, origin_h, origin_w, h, w):
    def scale_bbox(bbox, transform_h, transform_w, h, w):
        scale = min(h / transform_h, w / transform_w)
        resized_w = transform_w * scale
        resized_h = transform_h * scale
        bbox = [bbox[0] * w / resized_w, bbox[1] * h / resized_h,
                bbox[2] * w / resized_w, bbox[3] * h / resized_h]
        return bbox

    image_corners = np.array([origin_w, origin_h, origin_w, origin_h])
    bbox = scale_bbox(bbox, origin_h, origin_w, h, w)
    bbox = ([bbox[0], bbox[1], bbox[2], bbox[3]] * image_corners).astype(int)
    return bbox.tolist()


def resize_image(image, w, h):
    new_image = np.zeros((h, w, 3))
    scale = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)[:h, :w]
    new_image[:resized.shape[0], :resized.shape[1]] = resized
    return new_image


def preprocess_image_format(image, w, h, net_type):
    if net_type == 'face':
        image = resize_image(image, w, h)
    else:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    image = image.transpose((2, 0, 1))
    return image


def get_blur_coef(image):
    image = image.transpose((1, 2, 0)).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_coef = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_coef
