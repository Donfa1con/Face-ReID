VIDEO_PATH = '/video_data/origin'
RESULT_VIDEO_PATH = '/video_data/result'

INDEX_FACE_PATH = './index.faiss'

RABBITMQ = {
    'ip': 'rabbitmq',
    'port': '5672',
    'heartbeat': 60 * 60 * 2
}
DB = {
    'MONGODB': {
        'ip': 'mongodb',
        'port': 27017
    }
}

DIST_THRESHOLD = 0.6
FACE_CONF = 0.7
BLUR_THRESHOLD = 300
MIN_SIZE = 0.005
