FROM ubuntu:latest
WORKDIR home
COPY /load_balancer load_balancer
COPY /utils utils
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r load_balancer/requirements.txt
EXPOSE 12345
CMD ["python3", "load_balancer/server.py"]
