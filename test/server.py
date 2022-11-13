'''
# IP = socket.gethostname()
IP = ''
port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((IP, port))
s.listen()
while True:
    client_socket, client_ip = s.accept()
    print(f"{client_ip} connected")

    msg = "hello there"
    client_socket.send(bytes(msg, "utf-8"))

    client_socket.close()
    break
s.close()
'''

from api import LoadBalancer

server = LoadBalancer()
