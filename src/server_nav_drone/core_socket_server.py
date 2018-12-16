try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

import struct


class CoreSocketRequestHandler(SocketServer.StreamRequestHandler):
    def write_response(self, message):
        message_len = len(message)
        message = struct.pack("<i", message_len) + message
        self.wfile.write(message)

    def handle(self):
        while True:
            # get message length
            message_len_bits = self.rfile.read(4)
            message_len = struct.unpack("<i", message_len_bits)[0]
            message = self.rfile.read(message_len)
            response_writer = self.write_response
            self.server.process_message(message, response_writer)


class CoreSocketServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    def __init__(self, hostname, port, message_handler, server_fields):
        SocketServer.TCPServer.__init__(self, (hostname, port),
                                        CoreSocketRequestHandler,
                                        bind_and_activate=True)
        self.message_handler = message_handler
        self.message_handler.initialize_server(server_fields)

    def process_message(self, message, response_writer):
        self.message_handler.process_message(message, response_writer)


def launch_server(hostname, port, server_controller, server_fields):
    core_server = CoreSocketServer(hostname, port, server_controller, server_fields)
    core_server.serve_forever()
