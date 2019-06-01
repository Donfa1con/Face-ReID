import json
import os
import time

import pika
import pymongo

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
    filename = body.decode("utf-8")
    filename_path = os.path.join(config.VIDEO_PATH, filename)
    LOGGER.info('Checking: {0}'.format(filename))

    if os.path.isfile(filename_path):
        result = detect(DETECTION_CONFIG, filename_path)
        # if result:
        #     MONGO_CLIENT[config.DB['MONGODB']['db']][config.DB['MONGODB']['table']].update_one(
        #         {'_id': filename}, {'$set': result}, upsert=True
        #     )
        # else:
        #     response = {'error': 'File has not frames'}
        #     LOGGER.info(json.dumps(response))
    else:
        LOGGER.info('No such file: {0}'.format(filename_path))
    LOGGER.info('{0} DONE!'.format(filename_path))
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    while True:
        # try:
        #     MONGO_CLIENT = pymongo.MongoClient(
        #         host=config.DB['MONGODB']['ip'], port=config.DB['MONGODB']['port']
        #     )

            LOGGER.info('try person_detection consume')
            listen_queue(face)
            LOGGER.info('end person_detection consume')
            break
        #
        # except Exception as e:
        #     LOGGER.info('cannot start because {}'.format(e))
        #     time.sleep(5)
