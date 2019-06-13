import collections
import os

import cv2
import faiss
import numpy as np

import config
import utils

if os.path.isfile(config.INDEX_FACE_PATH):
    FAISS_INDEX = faiss.read_index(config.INDEX_FACE_PATH)
else:
    FAISS_INDEX = faiss.IndexFlatIP(256)


def get_face_emb(params, resized_face):
    output = list(params['reid']['net'].infer({"0": resized_face}).values())
    face_emb = np.array([v[0][0] for v in output[0][0]])
    norm_face_emb = (face_emb / np.linalg.norm(face_emb)).reshape(1, -1).astype('float32')
    return norm_face_emb


def get_faces(params, frame):
    p_idx = 2
    infer_frame = utils.preprocess_image_format(frame, **params['face']['input_size'],
                                                net_type='face')
    output = params['face']['net'].infer({'data': infer_frame})
    faces = [face for face in output['detection_out'][0][0] if face[p_idx] > config.FACE_CONF]
    return faces


def get_person_idx(face_emb, mongo, detected_face, padded_face, blur):
    if FAISS_INDEX.ntotal == 0:
        FAISS_INDEX.add(face_emb)
        person_name = 0
        create_mongo_face(person_name, mongo, detected_face, padded_face, blur)
    else:
        dist, idxs = FAISS_INDEX.search(face_emb, 1)
        dist = dist[0][0]
        idxs = idxs[0][0]

        if 1 - dist > config.DIST_THRESHOLD:
            person_name = FAISS_INDEX.ntotal
            FAISS_INDEX.add(face_emb)
            create_mongo_face(person_name, mongo, detected_face, padded_face, blur)
        else:
            person_name = idxs
    return person_name


def create_mongo_face(person_idx, mongo, detected_face, padded_face, blur):
    detected_face_string = cv2.imencode('.jpg', detected_face[..., ::-1])[1].tostring()
    padded_face_string = cv2.imencode('.jpg', padded_face[..., ::-1])[1].tostring()
    mongo[config.DB['MONGODB']['db']][config.DB['MONGODB']['table']].insert(
        {'_id': str(person_idx),
         'face_crop': padded_face_string,
         'face': detected_face_string,
         'blur': blur})


def update_mongo_faces(mongo, filename, faces_ts, detected_faces):
    for person_idx in detected_faces:
        mongo[config.DB['MONGODB']['db']][config.DB['MONGODB']['table']].update_one(
            {'_id': str(person_idx)},
            {'$set': {'paths.{0}'.format(filename.replace('.', '_')): faces_ts[person_idx]}},
            upsert=True
        )


def detect(params, filename, db_clients):
    mongo = db_clients['mongo']
    redis = db_clients['redis']
    redis.set(filename, 0)

    cap = cv2.VideoCapture(filename)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = 0
    faces_ts = collections.defaultdict(list)
    detected_faces = set()
    while cap.isOpened():
        ret, frame = cap.read()
        frame_num += 1
        if not ret:
            break

        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        faces = get_faces(params, frame)
        for face in faces:
            s = (face[6] - face[4]) * (face[5] - face[3])
            if s < config.MIN_SIZE:
                continue

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
                continue

            face_emb = get_face_emb(params, resized_face)

            y_pad = (ymax - ymin)
            x_pad = (xmax - xmin)
            padded_face = frame[max(0, ymin - y_pad):ymax + y_pad,
                                max(0, xmin - x_pad):xmax + x_pad]

            person_idx = get_person_idx(face_emb, mongo, origin_face, padded_face, blur)
            faces_ts[person_idx].append(ts)
            detected_faces.add(person_idx)

        if frame_num % (30 * 60) == 0:  # update db each 1 min of video
            redis.set(filename, frame_num / frame_count * 100)
            faiss.write_index(FAISS_INDEX, config.INDEX_FACE_PATH)
            update_mongo_faces(mongo, filename, faces_ts, detected_faces)
            detected_faces = set()

    update_mongo_faces(mongo, filename, faces_ts, detected_faces)
    faiss.write_index(FAISS_INDEX, config.INDEX_FACE_PATH)
    db_clients['redis'].set(filename, 100)
