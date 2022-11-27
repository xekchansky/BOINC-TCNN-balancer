import configparser
import datetime
import os

from kafka import KafkaConsumer


class LogConsumer:
    def __init__(self, credentials_path='credentials.ini', topic='logs', logs_path='logs'):
        config = configparser.ConfigParser()
        config.read(credentials_path)
        self.consumer = KafkaConsumer(
            'logs',
            bootstrap_servers=config['KAFKA']['addr'],
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username=config['KAFKA']['login'],
            sasl_plain_password=config['KAFKA']['password'],
            ssl_cafile="YandexInternalRootCA.crt")

        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = 'KAFKA_' + now + '.txt'
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        self.file_path = os.path.join(logs_path, filename)
        with open(self.file_path, 'w') as f:
            f.write('START' + now)

    def listen(self):
        for msg in self.consumer:
            msg_str = msg.key.decode("utf-8") + ":" + msg.value.decode("utf-8")
            with open(self.file_path, 'a') as f:
                f.write(f'\n{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} {msg_str}')


def main():
    listener = LogConsumer()
    listener.listen()


if __name__ == '__main__':
    main()
