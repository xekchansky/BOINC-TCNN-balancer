import argparse
import json
import threading

from app.client import NodeAPI


class AdminUI:
    def __init__(self, admin_node):
        self.admin_node = admin_node
        self.commands = {
            'GET MEMBERS': self.get_members,
            'START': self.start,
            'STOP': self.stop,
        }

    def run(self):
        commands_str, commands_dict = self.compile_commands()
        inp = '123'
        while inp != '':
            inp = input(f'\n{"-"*30}\n{commands_str}\n\nEnter command: ')
            if inp in commands_dict.keys():
                self.commands[commands_dict[inp]]()
            elif inp in self.commands.keys():
                self.commands[inp]()
            else:
                print('UNKNOWN COMMAND')

    def compile_commands(self):
        command_names = self.commands.keys()
        res_dict = {str(num): command for num, command in enumerate(command_names)}
        res = '\n'.join((' : '.join((str(num), command)) for num, command in enumerate(command_names)))
        return res, res_dict

    def get_members(self):
        members = self.admin_node.known_nodes_addr
        print('Total members: ', len(members))
        print('Members: ')
        for member in members:
            print(member)

    def start(self):
        self.admin_node.send_message(msg_type='START', msg=b'', target_node=self.admin_node.load_balancer)

    def stop(self):
        self.admin_node.send_message(msg_type='STOP', msg=b'', target_node=self.admin_node.load_balancer)


class AdminNode(NodeAPI):
    def __init__(self, ip='localhost', port=12345, logger=None):
        super().__init__(ip, port, logger)

        admin_node_msg_types = {
            'ADMIN_ACCEPTED': self.admin_accepted,
            'ADMIN_REJECTED': self.admin_rejected,
        }
        self.msg_types.update(admin_node_msg_types)

    def run(self):
        print('CONNECTING TO LOAD BALANCER')
        self.load_balancer.socket.connect(self.load_balancer.addr)
        print('CONNECTED')
        self.spawn_listener(self.load_balancer)
        self.admin_auth(admin_password='12345678')
        self.wait_for_threads()

    def admin_auth(self, admin_password):
        print('AUTHENTICATING')
        self.send_message(msg_type='ADMIN_CONNECT',
                          msg=admin_password.encode(self.encoding),
                          target_node=self.load_balancer)

    def admin_accepted(self, *_, **__):
        print('ADMIN CONNECTION ACCEPTED')
        terminal = AdminUI(self)
        thread = threading.Thread(target=terminal.run, args=())
        thread.daemon = True
        thread.start()

    def admin_rejected(self, *_, **__):
        print('ADMIN CONNECTION REJECTED')
        self.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default=None)
    parser.add_argument('--port', type=int, required=False, default=12345)
    return parser.parse_args()


def main(ip, port):
    if ip is None:
        with open('terraform_output.json') as f:
            ip = json.load(f)['external_ip_address_load_balancer']['value']
    AdminNode(ip=ip, port=port).run()


if __name__ == "__main__":
    args = parse_args()
    main(args.ip, args.port)
