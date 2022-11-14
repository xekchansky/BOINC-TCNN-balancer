FROM ubuntu:latest
COPY . .
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 12345
CMD ["python3", "server.py"]
