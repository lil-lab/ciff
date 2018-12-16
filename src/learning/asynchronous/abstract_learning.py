import torch


class AbstractLearning:

    def __init__(self, shared_model, local_model, calc_loss, optimizer, config, constants, tensorboard=None):
        self.shared_navigator_model = shared_model
        self.local_navigator_model = local_model
        self.config = config
        self.constants = constants
        self.calc_loss = calc_loss
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.iter = 0
        self.grad_log_enable = False
        self.grad_log_iter = 200

    def do_update(self, batch_replay_items):

        loss = self.calc_loss(batch_replay_items)

        if loss is None:
            return 0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_navigator_model.get_parameters(), 40)

        self.ensure_shared_grads(self.local_navigator_model, self.shared_navigator_model)
        self.optimizer.step()

        if self.grad_log_enable:
            if self.iter % self.grad_log_iter == 0:
                self.write_grad_summaries()
        self.iter += 1

        loss = float(loss.data[0])

        return loss

    @staticmethod
    def ensure_shared_grads(local_model, shared_model):

        local_model_parameters = local_model.get_parameters()
        shared_model_parameters = shared_model.get_parameters()

        for param, shared_param in zip(local_model_parameters, shared_model_parameters):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def write_grad_summaries(self):
        if self.tensorboard is None:
            return
        named_params = self.local_navigator_model.get_named_parameters()
        for name, parameter in named_params:
            weights = parameter.data.cpu()
            mean_weight = torch.mean(torch.abs(weights))
            weights = weights.numpy()
            self.tensorboard.log_histogram("hist_" + name + "_data", weights, bins=100)
            self.tensorboard.log_scalar("mean_" + name + "_data", mean_weight)
            if parameter.grad is not None:
                grad = parameter.grad.data.cpu()
                mean_grad = torch.mean(torch.abs(grad))
                grad = grad.numpy()
                self.tensorboard.log_histogram("hist_" + name + "_grad", grad, bins=100)
                self.tensorboard.log_scalar("mean_" + name + "_grad", mean_grad)


