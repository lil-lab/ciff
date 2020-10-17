# from pushover import Client


CLIENT_TOKEN = None
API_TOKEN = None


class PushoverLogger(object):
    def __init__(self, experiment_name):
        return
        self.experiment_name = experiment_name
        andrew_client = Client(ANDREW_CLIENT_TOKEN, api_token=ANDREW_API_TOKEN)
        self.clients = [andrew_client]

    def log(self, message):
        return
        for client in self.clients:
            client.send_message(message, title=self.experiment_name)

