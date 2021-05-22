FROM python:3.7

ADD ./ /worker

RUN pip3 install -r /worker/requirements.txt --no-cache-dir

WORKDIR /worker
CMD python -u main.py
