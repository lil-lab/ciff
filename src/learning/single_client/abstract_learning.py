import torch


class AbstractLearning:

    def __init__(self, model, calc_loss, optimizer, config, constants, tensorboard=None):
        self.model = model
        self.config = config
        self.constants = constants
        self.calc_loss = calc_loss
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.iter = 0
        self.grad_log_iter = 200

    def do_update(self, batch_replay_items):
        return self.do_update_iterative(batch_replay_items)

    def do_update_iterative(self, batch_replay_items):

        sum_loss = 0
        num_chunks = 0
        total_items = len(batch_replay_items)

        for i in range(0, total_items, 32):

            chunk = batch_replay_items[i:i + 32]
            loss = self.calc_loss(chunk)

            if loss is None:
                continue

            num_chunks += 1

            # Optimize the model.
            self.optimizer.zero_grad()
            loss.backward(retain_variables=True)
            for param in self.model.get_parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-15.0, 15.0)
            self.optimizer.step()
            # if self.iter % self.grad_log_iter == 0:
            #     self.write_grad_summaries()
            self.iter += 1
            sum_loss += float(loss.data[0])

        avg_loss = sum_loss/float(max(1, num_chunks))

        return avg_loss

    def write_grad_summaries(self):
        if self.tensorboard is None:
            return
        named_params = self.model.get_named_parameters()
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


