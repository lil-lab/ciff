import os

from tensorboardX import SummaryWriter


class Tensorboard:

    def __init__(self, experiment, log_dir="tensorboard_logs"):
        experiment_name = experiment.split("/")[-1]
        save_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(save_dir)
        self.index_dict = dict()

    def log_scalar(self, name, value, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_scalar(name, value, index)

    def log_histogram(self, name, value, bins, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_histogram(name, value, index, bins)

    def log(self, cross_entropy, loss, reward):
        self.log_scalar("cross_entropy", cross_entropy)
        self.log_scalar("loss", loss)
        self.log_scalar("total_reward", reward)

    def log_squared_loss(self, loss):
        self.log_scalar("Squared_Loss", loss)

    def log_action_prediction_loss(self, loss):
        self.log_scalar("Action_Prediction_Loss", loss)

    def log_temporal_autoencoder_loss(self, loss):
        self.log_scalar("Temporal_Autoencoder_Loss", loss)

    def log_object_detection_loss(self, loss):
        self.log_scalar("Object_Detection_Negative_Log_Probability", loss)

    def log_factor_entropy_loss(self, loss):
        self.log_scalar("Factor_Entropy", loss)

    def log_loglikelihood_position(self, loss):
        self.log_scalar("Position_log_likelihood", loss)

    def log_train_error(self, train_error):
        self.log_scalar("train_error", train_error)

    def log_test_error(self, test_error):
        self.log_scalar("test_error", test_error)

    def log_all_train_errors(self, edit_distance_error, mean_closest_distance_error, stop_dist_error):
        self.log_scalar("train_edit_dist_error", edit_distance_error)
        self.log_scalar("train_mean_closest_dist_error", mean_closest_distance_error)
        self.log_scalar("train_stop_dist_error", stop_dist_error)

    def log_all_test_errors(self, edit_distance_error, mean_closest_distance_error, stop_dist_error):
        self.log_scalar("test_edit_dist_error", edit_distance_error)
        self.log_scalar("test_mean_closest_dist_error", mean_closest_distance_error)
        self.log_scalar("test_stop_dist_error", stop_dist_error)

    def log_test_dist_error(self, test_error):
        self.log_scalar("test_dist_error", test_error)
