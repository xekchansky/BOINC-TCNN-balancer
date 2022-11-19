import pickle
import socket
import sys
import threading
from time import sleep


class Node:
    __slots__ = ['addr', 'socket', 'send_msg_lock']

    def __init__(self, addr=None, s=None):
        self.addr = addr
        self.socket = s
        self.send_msg_lock = threading.Lock()

    def __del__(self):
        if self.send_msg_lock.locked():
            self.send_msg_lock.release()
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
        self.msg_types = {
            'STOP': self.stop,
            'PING': self.ack
        }

    def __del__(self):
        sys.exit()

    def stop(self, msg, sender):
        self.__del__()

    def wait_for_threads(self):
        while len(self.threads):
            for thread in list(self.threads):
                thread.join()
                self.threads.remove(thread)

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

    def send_message(self, msg_type, msg, target_node):
        msg = bytes(f'{len(msg):<{self.msg_header_size}}{msg_type:<{self.msg_type_size}}', self.encoding) + msg
        try:
            target_node.send_msg_lock.acquire()
            target_node.socket.sendall(msg)
            target_node.send_msg_lock.release()
            return True
        except socket.error:
            target_node.send_msg_lock.release()
            return False

    def receive_message(self, target_node):
        try:
            msg_header = target_node.socket.recv(self.msg_header_size).decode(self.encoding)
            if not len(msg_header):
                return False
            msg_len = int(msg_header)
            msg_type = target_node.socket.recv(self.msg_type_size).decode(self.encoding).strip()
            msg_data = target_node.socket.recv(msg_len)
            return {"type": msg_type, "data": msg_data}
        except socket.error as e:
            print(e)
            return False

    def process_messages(self, target_node):
        while True:
            msg = self.receive_message(target_node)
            if msg:
                if msg['type'] in self.msg_types.keys():
                    self.msg_types[msg['type']](msg=msg['data'], sender=target_node)
                else:
                    print('UNKNOWN MSG TYPE:', msg['type'], msg['data'])
                    print('known msg types:', self.msg_types.keys())
            else:
                break

    def send_ping(self, target_node):
        self.send_message(msg_type='PING', msg='', target_node=target_node)

    def ping(self, msg, sender):
        self.send_message(msg_type='ACK', msg=msg, target_node=sender)

    def ack(self, *_):
        pass


class LoadBalancerAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        load_balancer_msg_types = {
        }
        self.msg_types.update(load_balancer_msg_types)

        self.run_server()

    def __del__(self):
        super().__del__()

    def stop(self, msg, sender):
        for node in list(self.nodes):
            if not self.send_message('STOP', msg, node.socket):
                self.lost_connection(node)
        super().stop(msg, sender)

    def run_server(self, heartbeat_rate=10):
        self.load_balancer.socket.bind(self.load_balancer.addr)
        self.load_balancer.socket.listen()
        self.spawn_connection_accepter(self.load_balancer.socket)
        while True:
            sleep(heartbeat_rate)
            self.broadcast_members()  # works as ping_members

    def ping_members(self):
        for node in self.nodes:
            self.send_ping(node)

    def broadcast_members(self):
        msg = pickle.dumps([node.addr for node in self.nodes])
        for node in list(self.nodes):
            if not self.send_message('NODES', msg, node.socket):
                self.lost_connection(node)


class NodeAPI(API):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        self.known_nodes_addr = set()

        node_msg_types = {
            'NODES': self.update_known_nodes,
        }
        self.msg_types.update(node_msg_types)

        self.run_node()

    def __del__(self):
        super().__del__()

    def run_node(self):
        self.load_balancer.socket.connect(self.load_balancer.addr)
        self.spawn_listener(self.load_balancer)
        self.wait_for_threads()

    def update_known_nodes(self, msg, *_):
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
