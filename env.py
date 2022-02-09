import socket
server = None
client = None
class Socket:
    def __init__(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 999))
        server.listen(1)

    def recive(self):
        return ""

    def send(self, data):
        pass