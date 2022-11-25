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
        self.ready_nodes = []

        self.threads = []
        self.stopped = False
        self.logger = logger

        self.msg_header_size = 10
        self.msg_type_size = 15
        self.encoding = 'UTF-8'
        self.msg_types = {
            'STOP': self.stop,
            'PING': self.ping,
            'ACK': self.ack,
            'BIG_MSG_PART': self.process_big_message_part,
        }
        self.big_messages_buffer = {}

    def __del__(self):
        sys.exit()

    def stop(self, *_, **__):
        self.logger.info('STOPPING')
        self.stopped = True
        self.__del__()

    def wait_for_stop(self):
        while not self.stopped:
            sleep(1)

    def wait_for_threads(self):
        while len(self.threads):
            for thread in list(self.threads):
                thread.join()
                self.threads.remove(thread)

    def new_connection(self, connection):
        new_node = Node(connection[1], connection[0])
        self.logger.info('node connected: %s', new_node.addr)
        self.nodes.add(new_node)
        self.spawn_listener(new_node)

    def lost_connection(self, node):
        if node in self.nodes:
            self.logger.info('node disconnected: %s', node.addr)
            self.nodes.remove(node)

        if node in self.ready_nodes:
            self.logger.info('node disconnected: %s', node.addr)
            self.ready_nodes.remove(node)

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
        self.process_messages(node)
        self.lost_connection(node)
        sys.exit()

    def send_message(self, msg_type, msg, target_node):
        self.logger.debug('SENDING MESSAGE to %s: %s size: %s', target_node.addr, msg_type, len(msg))
        msg = bytes(f'{len(msg):<{self.msg_header_size}}{msg_type:<{self.msg_type_size}}', self.encoding) + msg
        try:
            target_node.send_msg_lock.acquire()
            target_node.socket.sendall(msg)
            target_node.send_msg_lock.release()
            return True
        except socket.error:
            target_node.send_msg_lock.release()
            return False

    def send_big_message(self, msg_type, msg, target_node, part_size=8000):
        self.logger.debug('SENDING BIG MESSAGE to %s: %s size: %s', target_node.addr, msg_type, len(msg))
        msg_size = len(msg)
        number_of_submessages = msg_size // part_size + bool(msg_size % part_size)
        for i in range(number_of_submessages):
            left = i * part_size
            right = min(left + part_size, msg_size)
            submsg = msg[left:right]
            sub_type = 'BIG_MSG_PART'
            submsg = bytes(f'{msg_type:<{self.msg_type_size}}{number_of_submessages:<{self.msg_type_size}}', self.encoding) + submsg
            if not self.send_message(msg_type=sub_type, msg=submsg, target_node=target_node):
                return False
        return True

    def process_big_message_part(self, msg, sender):
        msg_type = msg[:self.msg_type_size].decode(self.encoding).strip()
        number_of_submessages = int(msg[self.msg_type_size:2 * self.msg_type_size].decode(self.encoding).strip())
        msg_part = msg[2 * self.msg_type_size:]
        key = (sender, msg_type)
        if key in self.big_messages_buffer.keys():
            self.big_messages_buffer[key][1].append(msg_part)
        else:
            self.big_messages_buffer[key] = [number_of_submessages, [msg_part, ]]
        if len(self.big_messages_buffer[key][1]) == self.big_messages_buffer[key][0]:
            self.msg_types[msg_type](msg=b''.join(self.big_messages_buffer[key][1]), sender=sender)
            del self.big_messages_buffer[key]

    def receive_message(self, target_node):
        try:
            msg_header = target_node.socket.recv(self.msg_header_size).decode(self.encoding)
            if not len(msg_header):
                return False
            msg_len = int(msg_header)
            msg_type = target_node.socket.recv(self.msg_type_size).decode(self.encoding).strip()
            msg_data = b''
            while len(msg_data) < msg_len:
                left_size = msg_len - len(msg_data)
                msg_data += target_node.socket.recv(left_size)
            return {"type": msg_type, "data": msg_data}
        except socket.error as e:
            self.logger.error("ERROR on socket %s: %s", target_node.addr, e)
            return False

    def process_messages(self, target_node):
        while True:
            msg = self.receive_message(target_node)
            if msg:
                self.logger.debug('RECEIVED MESSAGE from %s: %s size: %s',
                                  target_node.addr,
                                  msg['type'],
                                  len(msg['data']))
                if msg['type'] in self.msg_types.keys():
                    self.msg_types[msg['type']](msg=msg['data'], sender=target_node)
                else:
                    self.logger.error("ERROR UNKNOWN MSG TYPE: %s", msg['type'])
            else:
                break

    def send_ping(self, target_node):
        if not self.send_message(msg_type='PING', msg='', target_node=target_node):
            self.lost_connection(target_node)

    def ping(self, msg, sender):
        if not self.send_message(msg_type='ACK', msg=msg, target_node=sender):
            self.lost_connection(sender)

    def ack(self, *_, **__):
        pass
