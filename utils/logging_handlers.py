# from kafka import KafkaProducer
import datetime
import logging
import os.path

'''
class KafkaLoggingHandler(logging.Handler):

    def __init__(self, host, port, topic, key=None):
        logging.Handler.__init__(self)
        self.kafka_client = KafkaClient(host, port)
        self.key = key
        if key is None:
            self.producer = SimpleProducer(self.kafka_client, topic)
        else:
            self.producer = KeyedProducer(self.kafka_client, topic)

    def emit(self, record):
        # drop kafka logging to avoid infinite recursion
        if record.name == 'kafka':
            return
        try:
            # use default formatting
            msg = self.format(record)
            # produce message
            if self.key is None:
                self.producer.send_messages(msg)
            else:
                self.producer.send(self.key, msg)
        except:
            import traceback
            ei = sys.exc_info()
            traceback.print_exception(ei[0], ei[1], ei[2], None, sys.stderr)
            del ei

    def close(self):
        self.producer.stop()
        logging.Handler.close(self)
'''


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
        msg = self.format(record)
        with open(self.file_path, 'a') as f:
            f.write(f'\n{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} {msg}')
