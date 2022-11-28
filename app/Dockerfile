FROM ubuntu:latest
WORKDIR home
COPY /app app
COPY /utils utils
COPY /credentials.ini credentials.ini
COPY /YandexInternalRootCA.crt YandexInternalRootCA.crt
COPY /initial_model.pth initial_model.pth
COPY /terraform_output.json terraform_output.json
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
CMD ["python3", "app/client.py"]
