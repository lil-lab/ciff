import socket
import numpy as np
import scipy.misc


class ReliableConnect:
    DELIMITER = "<EOF>"
    BUFFER_SIZE = 1024
    ip_address = None
    port = None
    socket = None

    def __init__(self, ip_address, port, image_height, image_width):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.row = image_height
        self.col = image_width
        self.channel = 4
        self.id = 0
        self.connection = None
        self.total_bytes = self.row * self.col * self.channel * 4

    def connect(self):
        # create an INET, STREAMing socket
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((self.ip_address, self.port))
        except socket.error as msg:
            raise Exception('Bind failed. Error : ' + str(msg))

        self.socket.listen(10)

        # wait to accept a connection - blocking call
        # self.connection, addr = self.socket.accept()

    def initialize_server(self):
        self.connection, addr = self.socket.accept()

    def receive_image(self, save=False):
        """ Receives image over socket of size (ROW, COL, CHANNEL) """

        toread = self.total_bytes
        buf = bytearray(toread)
        view = memoryview(buf)

        self.id += 1

        while toread:
            nbytes = self.connection.recv_into(view, toread)
            view = view[nbytes:]  # slicing views is cheap
            toread -= nbytes
        img = np.frombuffer(buf, dtype='f4').reshape(
            (self.row, self.col, self.channel))

        # Remove alpha channel
        img = img[:, :, :3]
        if save:
            scipy.misc.imsave('./images/img' + str(self.id) + ".jpg", img)
        img = img.swapaxes(1, 2).swapaxes(0, 1)  # channel should be first for pytorch

        return img

    def send_message(self, message):
        if self.connection is None:
            raise Exception("Socket is not initialized. Please connect before use")

        self.connection.send((message + ReliableConnect.DELIMITER).encode())

    def receive_message(self):
        data = self.connection.recv(ReliableConnect.BUFFER_SIZE)
        data = str(data)
        data = data[len("Unity Manager: ") + 2:]
        return data

    def send_and_receive_message(self, message):
        self.send_and_receive_message(message)
        return self.receive_message()

    def close(self):
        self.connection.close()
        self.socket.close()
        self.connection = None
        self.socket = None
