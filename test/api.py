import pickle
import socket
import sys
import threading
from time import sleep


class API:
    def __init__(self, ip='localhost', port=12345, logger=None):
        self.load_balancer_addr = (ip, port)
        self.load_balancer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connections = {}
        self.threads = []
        self.logger = logger

        self.msg_header_size = 10
        self.msg_type_size = 5
        self.msg_types = {}

    def __del__(self):
        for thread in self.threads:
            thread.exit()
        self.load_balancer.close()

    def new_connection(self, connection):
        print('node connected', connection[1])
        self.connections[connection[0]] = connection[1]

    def lost_connection(self, connection_socket):
        if connection_socket in self.connections.keys():
            print('node disconnected', self.connections[connection_socket])
            del(self.connections[connection_socket])

    def wait_for_connections(self, target_socket):
        while True:
            self.new_connection(target_socket.accept())

    def spawn_connection_accepter(self, target_socket):
        self.threads.append(threading.Thread(target=self.wait_for_connections, args=(target_socket,)).start())

    def spawn_listener(self, target_socket):
        self.threads.append(threading.Thread(target=self.listener_routine, args=(target_socket,)).start())

    def listener_routine(self, target_socket):
        self.process_messages(target_socket)
        self.lost_connection(target_socket)
        sys.exit()

    def send_message(self, msg_type, msg, target):
        msg = bytes(f'{len(msg):<{self.msg_header_size}}{msg_type:<{self.msg_type_size}}', 'utf-8') + msg
        try:
            target.send(msg)
            return True
        except socket.error:
            return False

    def receive_message(self, target_socket):
        try:
            msg_header = target_socket.recv(self.msg_header_size).decode("utf-8")
            if not len(msg_header):
                return False
            msg_len = int(msg_header)
            msg_type = target_socket.recv(self.msg_type_size).decode("utf-8")
            msg_data = target_socket.recv(msg_len)
            return {"type": msg_type, "data": msg_data}
        except socket.error:
            return False

    def process_messages(self, target_socket):
        while True:
            msg = self.receive_message(target_socket)
            if msg:
                if msg['type'] in self.msg_types.keys():
                    self.msg_types[msg['type']](msg['data'])
                else:
                    print('UNKNOWN MSG TYPE:', msg['type'], msg['data'])
            else:
                break


class LoadBalancer(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)
        self.run_server()

    def __del__(self):
        super().__del__()
        for connection in self.connections.keys():
            connection[0].close()

    def run_server(self):
        self.load_balancer.bind(self.load_balancer_addr)
        self.load_balancer.listen()
        self.spawn_connection_accepter(self.load_balancer)
        while True:
            sleep(5)
            self.broadcast_members()

    def broadcast_members(self):
        msg = pickle.dumps([node for node in self.connections.values()])
        for connection_socket in list(self.connections.keys()):
            if not self.send_message('NODES', msg, connection_socket):
                self.lost_connection(connection_socket)


class Node(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.known_nodes = set()
        self.my_addr = None

        node_msg_types = {
            'ECHO': self.echo,
            'NODES': self.update_known_nodes,
        }
        self.msg_types.update(node_msg_types)

        self.run_node()

    def __del__(self):
        super().__del__()

    def run_node(self):
        self.load_balancer.connect(self.load_balancer_addr)
        self.process_messages(self.load_balancer)

    def echo(self, msg):
        self.my_addr = pickle.loads(msg)
        print(self.my_addr)

    def update_known_nodes(self, msg):
        nodes = pickle.loads(msg)
        for node in nodes:
            if node not in self.known_nodes:
                new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_socket.connect(node)
                self.connections[new_socket] = node
        self.known_nodes = set(nodes)
        print(self.known_nodes)
