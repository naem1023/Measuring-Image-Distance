# Image process service

## Start

### Run servers
```
$ docker-compose up -d --build rabbitmq api
```
### Run web
```
$ API_URI=xxxx docker-compose up -d --build web
```
### Run worker (cuda environment)
```
/worker $ python3 -m pip install -r ./requirements.txt
/worker $ FILE_SERVER=http://{API_HOST}:8080 \
  RABBITMQ_HOST={RABBIT_MQ_HOST} \
  QUEUE_NAME=IMAGE-PROCESS \
  RESULT_SUFFIX=_result \
  ACCESS_TOKEN=TOKEN_FOR_DIRECT_UPLOAD \
  DOWNLOAD_PATH=./files \
  python3 main.py
```

### API server
[API document](api/README.md)