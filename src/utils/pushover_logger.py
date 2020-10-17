# from pushover import Client


CLIENT_TOKEN = None
API_TOKEN = None


class PushoverLogger(object):
    def __init__(self, experiment_name):
        return # Comment out if using pushover
        self.experiment_name = experiment_name
        client = Client(CLIENT_TOKEN, api_token=API_TOKEN)
        self.clients = [client]

    def log(self, message):
        return
        for client in self.clients:
            client.send_message(message, title=self.experiment_name)

