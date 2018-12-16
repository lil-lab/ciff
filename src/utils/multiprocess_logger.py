import atexit
from multiprocessing import Process, Queue
import logging


class MultiprocessingLoggerManager(object):
    def __init__(self, file_path, logging_level):
        self.log_queue = Queue()
        self.p = Process(target=logger_daemon,
                    args=(self.log_queue, file_path, logging_level))
        self.p.start()
        atexit.register(self.cleanup)

    def get_logger(self, client_id):
        return MultiprocessingLogger(client_id, self.log_queue)

    def cleanup(self):
        self.p.terminate()


class MultiprocessingLogger(object):
    def __init__(self, client_id, log_queue):
        self.client_id = client_id
        self.log_queue = log_queue

    def log(self, message):
        self.log_queue.put("Client %r: %r" % (self.client_id, message))


def logger_daemon(log_queue, file_path, logging_level):
    logging.basicConfig(filename=file_path, level=logging_level)
    while True:
        logging.info(log_queue.get())