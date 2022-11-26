# from kafka import KafkaProducer
import datetime
import logging
import os.path

import configparser
from kafka import KafkaProducer


class KafkaLoggingHandler(logging.Handler):
    def __init__(self, credentials_path = 'credentials.ini', topic='logs', key=None):
        logging.Handler.__init__(self)

        config = configparser.ConfigParser()
        config.read(credentials_path)
        self.producer = KafkaProducer(
            bootstrap_servers=config['KAFKA']['addr'],
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username=config['KAFKA']['login'],
            sasl_plain_password=config['KAFKA']['password'],
            ssl_cafile="YandexInternalRootCA.crt")

    def emit(self, record):
        # drop kafka logging to avoid infinite recursion
        if record.name == 'kafka':
            return
        self.producer.send('logs', str(record).encode('utf-8'), b'key')
        self.producer.flush()

    def close(self):
        self.producer.close()
        logging.Handler.close(self)


class LocalHandler(logging.Handler):

    def __init__(self, logs_path):
        super().__init__()
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = now + '.txt'
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        self.file_path = os.path.join(logs_path, filename)
        with open(self.file_path, 'w') as f:
            f.write('START' + now)

    def emit(self, record):
        if 'botocore' in record.name:
            return
        msg = self.format(record)
        with open(self.file_path, 'a') as f:
            f.write(f'\n{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} {msg}')
