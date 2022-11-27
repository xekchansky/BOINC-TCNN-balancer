FROM ubuntu:latest
WORKDIR home
COPY /app app
COPY /utils utils
COPY /credentials.ini credentials.ini
COPY /initial_model.pth initial_model.pth
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r load_balancer/requirements.txt
EXPOSE 12345
CMD ["python3", "app/client.py"]
