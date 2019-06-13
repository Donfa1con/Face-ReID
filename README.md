# Face-ReID

Quick start

- update thresholds for faces into config.py

- update volume paths in docker-compse.yml

- build project
```docker
docker-compose up --build
```
- send path to video or image to Rebbit queue

```python
from pathlib import Path

VIDEO_PATH = '~/Desktop/driveData/' # mounted folder with video

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', port=5672)
)
channel = connection.channel()
channel.queue_declare(queue='face')
for filepath in Path(VIDEO_PATH).glob('**/*.avi'):
    channel.basic_publish(exchange='', routing_key='face',
                          body=str(filepath)[len(VIDEO_PATH):])
connection.close()
```

- connect to Redis to check progress
```python
import redis
import config

redis.Redis(host='localhost', port=6379, db=0)
for key in r.scan_iter("*"):
    if r.get(key) != 100:
        print(key, r.get(key))
```

- connect to Mongo to see face data
```python
import cv2
import numpy as np
import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')
face_data_0 = list(client.video_db.Faces.find({'_id': '0'}))[0]
 
# {'_id': 0,
#  'face': encoded image
#  'face_crop': padded encoded image
#  'paths': {'path_to_video_1': [40, 80], # millisec ts
#            'path_to_video_2': [120, 160, 200]}

face_img = cv2.imdecode(np.frombuffer(face_data_0['face_crop'], np.uint8), 1)
```

- connect to faiss to get similar persons

```python
import faiss

n_closest = 10
faiss_index = faiss.read_index('/path/to/faiss/face_vectors.faiss')
face_2_vecor = faiss_index.reconstruct(2)
product_distance, closest_person_ids = faiss_index.search(face_2_vecor.reshape(1, -1), n_closest)
```
Openvino Pretrained models
 - [Detectiion](https://docs.openvinotoolkit.org/latest/_face_detection_adas_0001_description_face_detection_adas_0001.html)
 - [Re-Identification](https://docs.openvinotoolkit.org/latest/_face_reidentification_retail_0095_description_face_reidentification_retail_0095.html)
 
 
