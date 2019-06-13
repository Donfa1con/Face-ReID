import json
import os
import time

import pika
import pymongo
import redis

import config
from face_reid import detect
from utils import create_logger, init_detection_config

DETECTION_CONFIG = init_detection_config()
LOGGER = create_logger()


def listen_queue(callback, queue_name='face'):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=config.RABBITMQ['ip'], port=config.RABBITMQ['port'],
                                  heartbeat=config.RABBITMQ['heartbeat'])
    )
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name,
                          on_message_callback=callback)
    LOGGER.info('start {0} consume'.format(queue_name))
    channel.start_consuming()


def face(ch, method, properties, body):
    filename = body.decode('utf-8')
    filename_path = os.path.join(config.VIDEO_PATH, filename)
    LOGGER.info('Checking: {0}'.format(filename_path))

    if int(REDIS_CLIENT.get(filename_path) or 0) == 100:
        LOGGER.info('File was cached in redis: {0}'.format(filename_path))
    elif os.path.isfile(filename_path):
        detect(DETECTION_CONFIG, filename_path, DB_CLIENTS)
    else:
        LOGGER.info('No such file: {0}'.format(filename_path))
    LOGGER.info('{0} DONE!'.format(filename_path))
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    while True:
        # try:
        MONGO_CLIENT = pymongo.MongoClient(
            host=config.DB['MONGODB']['ip'], port=config.DB['MONGODB']['port']
        )
        REDIS_CLIENT = redis.Redis(
            host=config.DB['REDIS']['ip'],
            port=config.DB['REDIS']['port'],
            db=config.DB['REDIS']['db']
        )
        DB_CLIENTS = {'redis': REDIS_CLIENT, 'mongo': MONGO_CLIENT}

        LOGGER.info('try person_detection consume')
        listen_queue(face)
        LOGGER.info('end person_detection consume')
        break

    # except Exception as e:
    #     LOGGER.info('cannot start because {}'.format(e))
    #     time.sleep(5)
