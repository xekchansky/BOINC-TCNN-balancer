import argparse
import hashlib
import logging
import pathlib
import pickle
import sys
from time import sleep

from data_distributor import DataDistributor

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import API
from utils.logging_handlers import LocalHandler


def get_fingerprint(string):
    return hashlib.md5(string.encode())


class LoadBalancerAPI(API):
    def __init__(self, ip='localhost', port=12345, admin_password='12345678', logger=None):
        super().__init__(ip, port, logger)

        load_balancer_msg_types = {
            'DATASET_REQUEST': self.send_dataset,
            'TEST_REQUEST': self.send_test_dataset,
            'ADMIN_CONNECT': self.admin_auth,
            'START': self.start,
        }
        self.msg_types.update(load_balancer_msg_types)

        self.admin_fingerprint = get_fingerprint(admin_password)
        self.non_auth_admin = None
        self.admin = None

        self.ds = DataDistributor(members_estimate=150)

    def __del__(self):
        super().__del__()

    def stop(self, msg, *_, **__):
        for node in list(self.nodes):
            if not self.send_message('STOP', msg, node):
                self.lost_connection(node)
        if self.admin is not None:
            del self.admin
            self.admin = None
        super().stop(msg)

    def run(self, heartbeat_rate=10):
        self.load_balancer.socket.bind(self.load_balancer.addr)
        self.load_balancer.socket.listen()
        self.spawn_connection_accepter(self.load_balancer.socket)
        while True:
            sleep(heartbeat_rate)
            self.broadcast_members()  # works as ping_members

    def send_dataset(self, sender, *_, **__):
        x_train, y_train = self.ds.get_data_part()
        msg = pickle.dumps((x_train, y_train))
        self.send_message(msg_type='DATASET', msg=msg, target_node=sender)

    def send_test_dataset(self, sender, *_, **__):
        x_test, y_test = self.ds.get_test_data()
        msg = pickle.dumps((x_test, y_test))
        self.send_message(msg_type='TEST_DATASET', msg=msg, target_node=sender)

    def admin_auth(self, msg, sender):
        self.non_auth_admin = sender
        self.nodes.remove(sender)
        password = msg.decode(self.encoding)
        if self.admin_fingerprint.hexdigest() == get_fingerprint(password).hexdigest():
            self.admin = sender
            self.non_auth_admin = None
            self.send_message(msg_type='ADMIN_ACCEPTED', msg=b'', target_node=sender)
        else:
            self.send_message(msg_type='ADMIN_REJECTED', msg=b'', target_node=sender)

    def start(self, msg, *_, **__):
        for node in list(self.nodes):
            if not self.send_message('START', msg, node):
                self.lost_connection(node)

    def ping_members(self):
        for node in self.nodes:
            self.send_ping(node)

    def broadcast_members(self):
        node_addr_list = [node.addr for node in self.nodes]
        for i, node in enumerate(list(self.nodes)):
            msg = pickle.dumps((i, node_addr_list))
            if not self.send_message('NODES', msg, node):
                self.lost_connection(node)

        if self.admin is not None:
            print('sending members to admin')
            msg = pickle.dumps((-1, node_addr_list))
            if not self.send_message('NODES', msg, self.admin):
                self.admin = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default='')
    parser.add_argument('--port', type=int, required=False, default=12345)
    parser.add_argument('--admin_password', type=str, required=False, default='12345678')
    return parser.parse_args()


def main(ip, port, admin_password):
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(LocalHandler('logs'))

    LoadBalancerAPI(ip=ip, port=port, admin_password=admin_password, logger=logger).run()


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port, args.admin_password)
