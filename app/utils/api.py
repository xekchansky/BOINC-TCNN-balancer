import pickle
import socket
import sys
import threading
from time import sleep


class Node:
    __slots__ = ['addr', 'socket']

    def __init__(self, addr=None, s=None):
        self.addr = addr
        self.socket = s

    def __del__(self):
        self.socket.close()


class API:
    def __init__(self, ip='localhost', port=12345, logger=None):
        self.load_balancer = Node((ip, port), socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        self.nodes = set()
        self.threads = []
        self.logger = logger

        self.msg_header_size = 10
        self.msg_type_size = 5
        self.encoding = 'UTF-8'
        self.msg_types = {}

    def __del__(self):
        sys.exit()

    def wait_for_threads(self):
        for thread in self.threads:
            thread.join()

    def new_connection(self, connection):
        new_node = Node(connection[1], connection[0])
        print('node connected', new_node.addr)
        self.nodes.add(new_node)
        self.spawn_listener(new_node)

    def lost_connection(self, node):
        if node in self.nodes:
            print('node disconnected', node.addr)
            self.nodes.remove(node)

    def spawn_connection_accepter(self, target_socket):
        thread = threading.Thread(target=self.wait_for_connections, args=(target_socket,))
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def wait_for_connections(self, target_socket):
        while True:
            self.new_connection(target_socket.accept())

    def spawn_listener(self, node):
        thread = threading.Thread(target=self.listener_routine, args=(node,))
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def listener_routine(self, node):
        self.process_messages(node.socket)
        self.lost_connection(node)
        sys.exit()

    def send_message(self, msg_type, msg, target_socket):
        msg = bytes(f'{len(msg):<{self.msg_header_size}}{msg_type:<{self.msg_type_size}}', self.encoding) + msg
        try:
            target_socket.sendall(msg)
            return True
        except socket.error:
            return False

    def receive_message(self, target_socket):
        try:
            msg_header = target_socket.recv(self.msg_header_size).decode(self.encoding)
            if not len(msg_header):
                return False
            msg_len = int(msg_header)
            msg_type = target_socket.recv(self.msg_type_size).decode(self.encoding).strip()
            msg_data = target_socket.recv(msg_len)
            return {"type": msg_type, "data": msg_data}
        except socket.error as e:
            print(e)
            return False

    def process_messages(self, target_socket):
        while True:
            msg = self.receive_message(target_socket)
            if msg:
                if msg['type'] in self.msg_types.keys():
                    self.msg_types[msg['type']](msg['data'])
                else:
                    print('UNKNOWN MSG TYPE:', msg['type'], msg['data'])
                    print('known msg types:', self.msg_types.keys())
            else:
                break


class LoadBalancerAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        load_balancer_msg_types = {
            'APORT': self.add_acceptor_port,
        }
        self.msg_types.update(load_balancer_msg_types)

        self.run_server()

    def __del__(self):
        super().__del__()

    def run_server(self):
        self.load_balancer.socket.bind(self.load_balancer.addr)
        self.load_balancer.socket.listen()
        self.spawn_connection_accepter(self.load_balancer.socket)
        while True:
            sleep(5)
            self.broadcast_members()

    def new_connection(self, connection):
        super().new_connection(connection)
        self.send_message("ECHO", pickle.dumps(connection[1]), connection[0])

    def broadcast_members(self):
        msg = pickle.dumps([node.addr for node in self.nodes])
        for node in list(self.nodes):
            if not self.send_message('NODES', msg, node.socket):
                self.lost_connection(node)

    def add_acceptor_port(self, msg):
        addr, port = pickle.loads(msg)
        for node in self.nodes:
            if node.addr == addr:
                node.addr = (node.addr[0], port)


class NodeAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.acceptor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.acceptor_socket.bind(('', 0))
        self.acceptor_port = self.acceptor_socket.getsockname()[1]

        self.known_nodes_addr = set()
        self.my_addr = None
        self.my_addr_for_lb = None

        node_msg_types = {
            'ECHO': self.echo,
            'HELLO': self.hello,
            'NODES': self.update_known_nodes,
            'STOP': self.stop,
        }
        self.msg_types.update(node_msg_types)

        self.run_node()

    def __del__(self):
        super().__del__()

    def run_node(self):
        self.load_balancer.socket.connect(self.load_balancer.addr)
        self.spawn_listener(self.load_balancer)

        while self.my_addr_for_lb is None:
            sleep(0.1)

        if self.my_addr_for_lb is not None:
            msg = pickle.dumps((self.my_addr_for_lb, self.acceptor_port))
            self.send_message('APORT', msg, self.load_balancer.socket)
            self.my_addr = (self.my_addr_for_lb[0], self.acceptor_port)

        # weak place ip of acceptor probably could be wrong
        self.acceptor_socket.listen()
        self.spawn_connection_accepter(self.acceptor_socket)
        self.wait_for_threads()

    def echo(self, msg):
        self.my_addr_for_lb = pickle.loads(msg)

    def hello(self, msg):
        print(msg.decode(self.encoding))

    def update_known_nodes(self, msg):
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
            if node_addr not in self.known_nodes_addr \
                    and node_addr != self.my_addr_for_lb \
                    and node_addr != self.my_addr:
                new_node = Node(node_addr, socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                print('my accept addr = ', self.acceptor_socket.getsockname())
                print('connecting to ', node_addr)
                new_node.socket.connect(node_addr)
                self.spawn_listener(new_node)
                self.nodes.add(new_node)
                self.known_nodes_addr.add(node_addr)
                self.send_message('HELLO', bytes(f'Hi from {self.my_addr}', self.encoding), new_node.socket)

    def stop(self, msg):
        msg.decode(self.encoding)
        self.__del__()
