VIDEO_PATH = '/video'

IMAGE_PATH = '/faces'

INDEX_FACE_PATH = '/faiss/face_vectors.faiss'
RABBITMQ = {
    'ip': 'rabbitmq',
    'port': '5672',
    'heartbeat': 0
}

DB = {
    'MONGODB': {
        'ip': 'mongodb',
        'port': 27017,
        'db': 'video_db',
        'table': 'Faces'
    },
    'REDIS': {
        'ip': 'redis',
        'port': 6379,
        'db': 0
    }
}

DIST_THRESHOLD = 0.4
FACE_CONF = 0.95
BLUR_THRESHOLD = 150
