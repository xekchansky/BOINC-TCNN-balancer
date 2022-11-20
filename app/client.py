import argparse
import json
import os
import pathlib
import pickle
import socket
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.api import API, Node


class NodeAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.known_nodes_addr = set()

        node_msg_types = {
            'NODES': self.update_known_nodes,
        }
        self.msg_types.update(node_msg_types)

    def __del__(self):
        super().__del__()

    def run(self):
        self.load_balancer.socket.connect(self.load_balancer.addr)
        self.spawn_listener(self.load_balancer)
        self.wait_for_threads()

    def update_known_nodes(self, msg, *_, **__):
        lb_nodes_addr = set(pickle.loads(msg))

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

    NodeAPI(ip=ip, port=port).run()


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port)
