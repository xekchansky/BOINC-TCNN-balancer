import socket

'''IP = 'localhost'
port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((IP, port))
print(f"Connected to {IP}:{port}")

while True:
    msg = s.recv(16)
    print(msg)
    break
s.close()
'''

from api import Node

IP = 'localhost'
port = 12345

client = Node(IP, port)