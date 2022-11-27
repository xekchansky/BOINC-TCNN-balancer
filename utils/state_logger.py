from time import time
import json


class StateLogger:
    def __init__(self, logger=None, send_period=10):
        self.logger = logger
        self.send_period = send_period
        self.last_log = time()
        self.last_state_start = time()
        self.current_state = 'START'
        self.statistics = {
            'START': 0,
            'BACKWARD': 0,
            'SYNC': 0,
            'WAIT': 0
        }

    def send_log(self):
        stat = json.dumps(self.statistics)
        self.logger.info(f'STATES TIME: {stat}')
        print('send_log', stat)

    def log(self, key):
        self.statistics[self.current_state] += time() - self.last_state_start
        if key != 'END':
            self.current_state = key
        if time() - self.last_log > self.send_period or key == 'END':
            self.last_log = time()
            self.send_log()
        self.last_state_start = time()

    def backward(self):
        self.log('BACKWARD')

    def sync(self):
        self.log('SYNC')

    def wait(self):
        self.log('WAIT')

    def close(self):
        self.log('END')
