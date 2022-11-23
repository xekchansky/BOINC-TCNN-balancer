import argparse
import json
import logging
import os
import pathlib
import pickle
import socket
import sys
import threading
from time import sleep, time

import boto3

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import API, Node
from utils.logging_handlers import LocalHandler


class NodeAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.connected_nodes_addr = set()
        self.ready_nodes_addr = []
        self.id = None

        node_msg_types = {
            'NODES': self.update_known_nodes,
            'START': self.start,
            'DATASET': self.download_dataset,
            'SUBMIT': self.income_submit,
        }
        self.msg_types.update(node_msg_types)

        self.data_path = 'data'

    def __del__(self):
        super().__del__()

    def run(self):
        self.load_balancer.socket.connect(self.load_balancer.addr)
        self.spawn_listener(self.load_balancer)
        self.request_dataset()
        self.wait_for_stop()

    def request_dataset(self):
        """Request filenames for local training dataset"""
        self.send_message(msg_type='DATASET_REQUEST', msg=b'', target_node=self.load_balancer)

    def download_dataset(self, msg, *_, **__):
        """Download of dataset images by received filenames"""
        x_train, y_train = pickle.loads(msg)

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        s3 = boto3.client('s3', endpoint_url='https://storage.yandexcloud.net')
        for file_name in x_train:
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    s3.download_fileobj('modified-kylberg-dataset', os.path.join('dataset', file_name), f)

        self.send_message(msg_type='READY', msg=b'', target_node=self.load_balancer)

    def start(self, *_, **__):
        thread = threading.Thread(target=self.routine, args=())
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def routine(self):
        """Learning routine after 'START' message"""
        while True:
            start = time()
            sleep(2)  # do something
            elapsed_time = time() - start
            msg = pickle.dumps((self.id, elapsed_time))
            self.send_message(msg_type='SUBMIT', msg=msg, target_node=self.load_balancer)

    def income_submit(self, msg, *_, **__):
        node_id, elapsed_time = pickle.loads(msg)
        print(f'ME: {self.id} RECEIVED: {node_id} {elapsed_time}')

    def remove_node_by_addr(self, node_addr):
        for node in list(self.nodes):
            if node.addr == node_addr:
                self.nodes.remove(node)
                self.connected_nodes_addr.remove(node_addr)
        for node in list(self.ready_nodes):
            if node.addr == node_addr:
                self.ready_nodes.remove(node)
                self.ready_nodes_addr.remove(node_addr)

    def move_node_to_ready_by_addr(self, node_addr):
        for node in list(self.nodes):
            if node.addr == node_addr:
                self.nodes.remove(node)
                self.ready_nodes.append(node)
                self.connected_nodes_addr.remove(node_addr)
                self.ready_nodes_addr.append(node_addr)

    def update_known_nodes(self, msg, *_, **__):
        i, lb_nodes_addr, lb_ready_nodes_addr = pickle.loads(msg)
        self.id = i

        # remove disconnected nodes
        for node_addr in (list(self.connected_nodes_addr) + list(self.ready_nodes_addr)):
            if (node_addr not in lb_nodes_addr) and (node_addr not in lb_ready_nodes_addr):
                self.remove_node_by_addr(node_addr)

        # move to ready nodes
        for node_addr in list(self.connected_nodes_addr):
            if node_addr in lb_ready_nodes_addr:
                self.move_node_to_ready_by_addr(node_addr)

        # add new connected nodes
        for node_addr in lb_nodes_addr:
            if node_addr not in self.connected_nodes_addr:
                new_node = Node(node_addr, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                self.nodes.add(new_node)
                self.connected_nodes_addr.add(node_addr)

        # add new ready nodes
        for node_addr in lb_ready_nodes_addr:
            if node_addr not in self.ready_nodes_addr:
                new_node = Node(node_addr, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                self.ready_nodes.append(new_node)
                self.ready_nodes_addr.append(node_addr)


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
