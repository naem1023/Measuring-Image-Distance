import os
import random
import time
import string
from typing import List, Optional

import pika
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from aiofile import AIOFile, Writer, Reader

upload_directory = os.environ['UPLOAD_PATH']
access_token = os.environ['ACCESS_TOKEN']
result_suffix = os.environ['RESULT_SUFFIX']
queue_name = os.environ['QUEUE_NAME']
rabbitmq_host = os.environ['RABBITMQ_HOST']

try:
    os.mkdir(upload_directory)
    print(os.getcwd())
except Exception as e:
    print(e)


def random_string_with_time(length: int):
    return ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    ) + '-' + str(int(time.time()))


async def store_file(uploaded_file: UploadFile, destination: string):
    async with AIOFile(os.path.join(upload_directory, destination), 'wb') as f:
        writer = Writer(f)
        while True:
            chunk = await uploaded_file.read(8192)
            if not chunk:
                break
            await writer(chunk)
    return destination


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    result = []
    for uploaded_file in files:
        filename = uploaded_file.filename
        extension = filename.split('.')[-1]
        stored_filename = random_string_with_time(10) + '.' + extension
        await store_file(uploaded_file, stored_filename)

        channel.basic_publish(exchange='', routing_key=queue_name, body=stored_filename.encode())
        result.append({'original': filename, 'stored': stored_filename})
    connection.close()
    return {'files': result}


@app.post("/files/{stored_filename}")
async def upload_file(
    stored_filename: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    if authorization != f'Basic {access_token}':
        raise HTTPException(
            status_code=403,
            detail=f'Invalid token in header should Authorization: Basic XX'
        )
    filename = file.filename
    await store_file(file, stored_filename)
    return {'original': filename, 'stored': stored_filename}


@app.get("/files/{filename}")
async def get_file(filename: str):
    if not os.path.isfile(os.path.join(upload_directory, filename)):
        raise HTTPException(status_code=404, detail='file not found')

    return FileResponse(os.path.join(upload_directory, filename))


@app.get("/files/{filename}/:result")
async def get_result_of_file(filename: str):
    if not os.path.isfile(os.path.join(upload_directory, filename)):
        raise HTTPException(status_code=404, detail='file not found')

    result_filename = filename + result_suffix
    try:
        return FileResponse(os.path.join(upload_directory, result_filename))
    except FileNotFoundError:
        raise HTTPException(status_code=202, detail=f'now processing on {result_filename}')
