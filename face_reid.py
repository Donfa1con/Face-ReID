import os
import time

import cv2
import faiss
import numpy as np

import config
import utils

if os.path.isfile(config.INDEX_FACE_PATH):
    FAISS_INDEX = faiss.read_index(config.INDEX_FACE_PATH)
else:
    FAISS_INDEX = faiss.IndexFlatIP(256)


def detect(params, filename):
    cap = cv2.VideoCapture(os.path.join(config.VIDEO_PATH, filename))
    out = utils.init_result_writer(cap, os.path.join(config.RESULT_VIDEO_PATH,
                                                     filename[len(config.VIDEO_PATH) + 1:]))
    p_id = 2
    num = 0
    while cap.isOpened():
        start = time.time()
        num += 1

        ret, frame = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret:
            break

        infer_frame = utils.preprocess_image_format(frame, **params['face']['input_size'],
                                                    net_type='face')
        output = params['face']['net'].infer({'data': infer_frame})
        detection_out = [face for face in output['detection_out'][0][0]
                         if face[2] > config.FACE_CONF]
        dist = None
        for face in detection_out:
            skip = False
            person_name = None

            s = (face[6] - face[4]) * (face[5] - face[3])
            if s < config.MIN_SIZE:
                skip = True

            xmin, ymin, xmax, ymax = utils.get_face_rect(face[3:], *frame.shape[:2],
                                                         **params['face']['input_size'])
            origin_face = frame[ymin:ymax, xmin:xmax]

            if not origin_face.shape[0] or not origin_face.shape[1]:
                continue

            resized_face = utils.preprocess_image_format(origin_face,
                                                         **params['reid']['input_size'],
                                                         net_type='reid')

            blur = utils.get_blur_coef(resized_face)
            if blur > config.BLUR_THRESHOLD:
                skip = True

            output = list(params['reid']['net'].infer({"0": resized_face}).values())
            face_emb = np.array([v[0][0] for v in output[0][0]])
            norm_face_emb = (face_emb / np.linalg.norm(face_emb)).reshape(1, -1).astype('float32')

            if FAISS_INDEX.ntotal == 0 and not skip:
                FAISS_INDEX.add(norm_face_emb)
                person_name = 0
            else:
                dist, idxs = FAISS_INDEX.search(norm_face_emb, 1)
                dist = dist[0][0]
                idxs = idxs[0][0]

                if 1 - dist > config.DIST_THRESHOLD and not skip:
                    FAISS_INDEX.add(norm_face_emb)
                    person_name = FAISS_INDEX.ntotal
                else:
                    person_name = idxs

            if skip:
                color = (0, 0, 255)
                if person_name is None:
                    person_name = 'skip'
            else:
                color = (255, 255, 0)

            frame = utils.draw_face_rect(frame, xmin, ymin, xmax, ymax, face[p_id], color)

            cv2.putText(frame, "person {0}".format(person_name),
                        (xmin, ymin - 24), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            cv2.putText(frame, "s: {0:.4f}".format(s),
                        (xmin, ymin - 38), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            cv2.putText(frame, "blur: {0}".format(int(blur)),
                        (xmin, ymin - 52), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if dist is not None:
                cv2.putText(frame, "dist {0:.4f}".format(1 - dist),
                            (xmin, ymin - 64), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        end = time.time() - start
        cv2.putText(frame, str(round(1 / end, 2)) + 'fps',
                    (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

        out.write(frame)

    faiss.write_index(FAISS_INDEX, config.INDEX_FACE_PATH)
