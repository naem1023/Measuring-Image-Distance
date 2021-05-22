import os
import random
import string
import tempfile
import time
import urllib.request

import pika
import requests
from pika.adapters.blocking_connection import BlockingChannel

download_directory = os.environ['DOWNLOAD_PATH']
file_server = os.environ['FILE_SERVER']
queue_name = os.environ['QUEUE_NAME']
rabbitmq_host = os.environ['RABBITMQ_HOST']
access_token = os.environ['ACCESS_TOKEN']
result_suffix = os.environ['RESULT_SUFFIX']

try:
    os.mkdir(download_directory)
    print(os.getcwd())
except Exception as e:
    print(e)


def random_string_with_time(length: int):
    return ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    ) + '-' + str(int(time.time()))


def process(filename: str):
    local_filename = random_string_with_time(10)
    with urllib.request.urlopen(f'{file_server}/files/{filename}') as http:
        with open(os.path.join(download_directory, local_filename), 'wb') as f:
            f.write(http.read())

    print(os.listdir(download_directory))
    # TODO: Process image (local_filename)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        # TODO: Store processed result of image
        f.write(b'result')
        f.close()

        upload_filename = filename + result_suffix
        result = requests.post(
            f'{file_server}/files/{upload_filename}',
            files={'file': open(f.name, 'rb')},
            headers={'Authorization': f'Basic {access_token}'},
        )
        print(result)


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
    channel: BlockingChannel = connection.channel()

    channel.queue_declare(queue=queue_name)

    def callback(ch: BlockingChannel, method, properties, body: bytes):
        print(f'{body} received')
        process(body.decode())
        ch.basic_ack(delivery_tag=method.delivery_tag)

    print('working on message consuming')
    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    channel.start_consuming()


if __name__ == '__main__':
    main()
