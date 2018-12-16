import socket
import logging
import numpy as np
import scipy.misc


class ReliableConnect:
    """ A simple class for one to one socket communication """

    DELIMITER = "<EOF>"
    BUFFER_SIZE = 1024
    ip_address = None
    port = None
    socket = None
    connection = None

    def __init__(self, ip_address, port, image_row, image_col):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connection = None
        self.row = image_row
        self.col = image_col
        self.channel = 4
        self.id = 0
        self.total_bytes = self.row * self.col * self.channel * 4

    def connect(self):
        # create an INET, STREAMing socket
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind(("0.0.0.0", self.port))
        except socket.error as msg:
            raise Exception('Bind failed. Error : ' + str(msg))

        logging.info("Created the server at port " + str(self.port) + " and address localhost")
        self.socket.listen(10)
        # self.socket.connect(("localhost", self.port))
        logging.info("Server is listening")

        # wait to accept a connection - blocking call
        self.connection, addr = self.socket.accept()
        logging.info('Connected with %r : %r', addr[0], addr[1])

    def send_message(self, message):
        if self.connection is None:
            raise Exception("Socket is not initialized. Please connect before use")

        self.connection.send(message)

    def close(self):
        self.connection.close()
        self.socket.close()
        self.connection = None
        self.socket = None

    def receive_message(self):
        data = self.connection.recv(ReliableConnect.BUFFER_SIZE)
        return data

    def receive_image(self):
        """ Receives image over socket of size (ROW, COL, CHANNEL) """

        toread = self.total_bytes
        buf = bytearray(toread)
        view = memoryview(buf)
        
        self.id += 1

        while toread:
            nbytes = self.connection.recv_into(view, toread)
            view = view[nbytes:]  # slicing views is cheap
            toread -= nbytes
        img = np.frombuffer(buf, dtype='f4').reshape((self.row, self.col, self.channel))

        # Remove alpha channel
        img = img[:, :, :3]
        img = np.fliplr(np.rot90(img, k=2))

        # scipy.misc.imsave('./attention_prob/received_image_' + str(self.id) + ".jpg", img)

        img = img.swapaxes(1, 2).swapaxes(0, 1)

        return img

    def send_and_receive_message(self, message):
        self.send_message(message)
        return self.receive_message()
