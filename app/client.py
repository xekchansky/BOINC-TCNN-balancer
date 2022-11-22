import argparse
import json
import logging
import os
import pathlib
import pickle
import socket
import sys
import threading
from time import sleep

import boto3

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import API, Node
from utils.logging_handlers import LocalHandler


class NodeAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.known_nodes_addr = set()
        self.id = None

        node_msg_types = {
            'NODES': self.update_known_nodes,
            'START': self.start,
            'DATASET': self.download_dataset,
        }
        self.msg_types.update(node_msg_types)

        self.data_path = 'data'

    def __del__(self):
        super().__del__()

    def run(self):
        self.load_balancer.socket.connect(self.load_balancer.addr)
        self.spawn_listener(self.load_balancer)
        self.request_dataset()
        self.wait_for_threads()

    def request_dataset(self):
        self.send_message(msg_type='DATASET_REQUEST', msg=b'', target_node=self.load_balancer)

    def download_dataset(self, msg, *_, **__):
        x_train, y_train = pickle.loads(msg)
        print('downloading objects:', len(x_train))

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        s3 = boto3.client('s3', endpoint_url='https://storage.yandexcloud.net')
        for file_name in x_train:
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    s3.download_fileobj('modified-kylberg-dataset', os.path.join('dataset', file_name), f)

        print('objects downloaded')

    def start(self, *_, **__):
        thread = threading.Thread(target=self.routine, args=())
        thread.daemon = True
        thread.start()

    def routine(self):
        while True:
            sleep(2)
            print(self.id)

    def update_known_nodes(self, msg, *_, **__):
        i, lb_nodes_addr = pickle.loads(msg)
        self.id = i

        # remove disconnected nodes
        for node_addr in list(self.known_nodes_addr):
            if node_addr not in lb_nodes_addr:
                for node in list(self.nodes):
                    if node.addr == node_addr:
                        self.nodes.remove(node)
                        self.known_nodes_addr.remove(node_addr)

        # add new nodes
        for node_addr in lb_nodes_addr:
            if node_addr not in self.known_nodes_addr:
                new_node = Node(node_addr, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                self.nodes.add(new_node)
                self.known_nodes_addr.add(node_addr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default=None)
    parser.add_argument('--port', type=int, required=False, default=12345)
    return parser.parse_args()


def main(ip, port):
    if ip is None:
        output_path = os.path.join(str(pathlib.Path(__file__).parent.parent.resolve()), 'terraform_output.json')
        with open(output_path) as f:
            ip = json.load(f)['external_ip_address_load_balancer']['value']

    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(LocalHandler('logs'))

    NodeAPI(ip=ip, port=port, logger=logger).run()


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port)
