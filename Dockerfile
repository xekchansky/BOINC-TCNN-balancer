FROM horovod/horovod-cpu

WORKDIR "/home/diplom"

COPY app/* app/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        unzip

RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir imageio
RUN pip install --no-cache-dir sklearn
RUN pip install --no-cache-dir torch-summary
RUN pip install --no-cache-dir google-api-python-client google-auth-httplib2 google-auth-oauthlib

