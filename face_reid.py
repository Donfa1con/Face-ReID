import os
import time

import cv2
import numpy as np
from scipy.spatial import distance

import config
import utils


def update_face_data(face_data, new_person_num, ts, face_emb):
    cosine_dst = np.vectorize(distance.cosine, signature='(n),(n)->()')
    if not face_data:
        face_data.append([new_person_num, str(ts), face_emb])
    else:
        dst_results = cosine_dst([face_emb], np.array(face_data)[:, 2].tolist())
        best_match_idx = np.argmin(dst_results)
        if dst_results[best_match_idx] > 0.4:
            face_data.append([new_person_num + 1, str(ts), face_emb])
        else:
            face_data[best_match_idx][1] = '{0},{1}'.format(face_data[best_match_idx][1], ts)
            face_data[best_match_idx][2] = np.mean([face_emb, face_data[best_match_idx][2]], axis=0)
    return face_data


def detect(params, filename):
    cap = cv2.VideoCapture(os.path.join(config.VIDEO_PATH, filename))
    out = utils.init_result_writer(cap, os.path.join(config.RESULT_VIDEO_PATH, filename))

    p_id = 2
    while cap.isOpened():
        start = time.time()

        ret, frame = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret:
            break

        infer_frame = utils.preprocess_image_format(frame, **params['face']['input_size'])
        output = params['face']['net'].infer({'data': infer_frame})
        detection_out = [face for face in output['detection_out'][0][0]
                         if face[2] > config.FACE_CONF]
        outputs = []
        for face in detection_out:
            xmin, ymin, xmax, ymax = utils.get_face_rect(*face[3:], *frame.shape[:2])

            origin_face = frame[ymin:ymax, xmin:xmax]
            resized_face = utils.preprocess_image_format(origin_face,
                                                         **params['face']['input_size'])

            if utils.get_blur_coef(resized_face) > config.BLUR_THRESHOLD:
                continue

            output = list(params['face']['net'].infer({0: resized_face}).values())
            face_emb = np.array([v[0][0] for v in output[0][0]])
            outputs.append(face_emb)

            frame = utils.draw_face_rect(frame, xmin, ymin, xmax, ymax, face[p_id])

            cv2.putText(frame, "person -1",
                        (xmin, ymin - 24), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

        end = time.time() - start
        cv2.putText(frame, str(round(1 / end, 2)) + 'fps',
                    (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

        out.write(frame)
