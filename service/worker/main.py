import os
import random
import string
import tempfile
import time
import urllib.request

from skimage import io
import torch
import h5py
import numpy as np
import pika
import requests
from pika.adapters.blocking_connection import BlockingChannel

import predict.predictor.merger

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


def generate_result(image_path: str) -> str:
    # Predict distance of all pixels
    merger = predictor.merger.Merger('predict/model_state_dict.pth')
    predict = merger.merge(image_path)

    # Transform depth array to rgb array
    predict_img = predictor.merger.transform_to_rgb(predict)

    # Compare prediction and real distance
    # TODO: Apply labeled data path
    path_to_depth_v1 = '/nyu_data/nyu_depth_data_labeled.mat'
    f = h5py.File(path_to_depth_v1)

    real_distance = depth = f['depths'][0]
    real_distance = predictor.merger.transpose_img(real_distance)
    rmse = predictor.merger.RMSELoss()

    rmse_error = rmse(torch.tensor(predict), torch.tensor(real_distance))
    print('rmse_error =', rmse_error.item())

    result_filename = 'demo_predict_distance.jpg'
    io.imsave(result_filename, predict_img)
    return result_filename


def process(filename: str):
    local_filename = random_string_with_time(10)
    with urllib.request.urlopen(f'{file_server}/files/{filename}') as http:
        with open(os.path.join(download_directory, local_filename), 'wb') as f:
            f.write(http.read())

    print(os.listdir(download_directory))
    result_file = generate_result(os.path.join(download_directory, local_filename))

    upload_filename = filename + result_suffix
    result = requests.post(
        f'{file_server}/files/{upload_filename}',
        files={'file': open(result_file, 'rb')},
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
