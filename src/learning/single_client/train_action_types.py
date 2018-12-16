import logging
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from utils.cuda import cuda_var


class TrainActionTypes:

    def __init__(self, model, text_embedding_model):
        self.model = model
        self.text_embedding_model = text_embedding_model
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.max_epoch = 20

    def test(self, test_dataset):

        precision = 0
        num_counts = 0
        for dp in test_dataset:
            """ Generate encoding """

            instruction_embedding = self.get_instruction_embedding(dp)
            token_ids = self.model.greedy_decoding(instruction_embedding)
            for token_id, gold_id in zip(token_ids, dp.gold_token_seq):

                token_id = token_id.data.cpu().numpy()[0]
                if token_id == gold_id:
                    precision += 1.0
                num_counts += 1.0

        precision = precision/float(max(1, num_counts))
        logging.info("Num counts %f and precision is %f " % (num_counts, precision))
        return precision

    def get_instruction_embedding(self, dp):

        instructions = [dp.instruction]
        instructions_batch = cuda_var(torch.from_numpy(np.array(instructions)).long())
        _, text_emb = self.text_embedding_model(instructions_batch)
        text_emb = Variable(text_emb.data)

        return text_emb

    def do_update(self, dp):

        loss = self.model.get_loss(instruction_embedding=self.get_instruction_embedding(dp),
                                   token_ids=dp.gold_token_seq)
        if loss is None:
            return

        self.optimizer.zero_grad()
        loss.backward(retain_variables=True)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-15.0, 15.0)
        self.optimizer.step()

        return loss

    @staticmethod
    def parse(house_id):

        lines = open("./simulators/house/datapoint_type_house%d.txt" % (house_id))
        dataset = []
        for line in lines:
            words = line.split()
            id_ = int(words[0])

            gold_token_seq = []
            for word in words[1:]:
                if word == "navigation":
                    gold_token_seq.append(0)
                elif word == "manipulation":
                    gold_token_seq.append(1)
                else:
                    raise AssertionError("Unknown word ", word)
            gold_token_seq.append(2)

            dp = TrainActionTypeDataPoint(instruction=None, instruction_string=None, id_=id_, gold_token_seq=gold_token_seq)
            dataset.append(dp)
        return dataset

    def do_train(self, train_dataset, tune_dataset, experiment_name, save_best_model=False):
        """ Perform training """

        dataset_size = len(train_dataset)

        # Test on tuning data with initialized model
        self.test(tune_dataset)

        for epoch in range(1, self.max_epoch + 1):

            logging.info("Starting epoch %d", epoch)

            for data_point_ix, data_point in enumerate(train_dataset):

                if (data_point_ix + 1) % 100 == 0:
                    logging.info("Done %d out of %d", data_point_ix, dataset_size)

                # Perform update
                loss_val = self.do_update(data_point)

            self.test(tune_dataset)

            # Save the model
            torch.save(self.model.state_dict(),
                       experiment_name + "/goal_prediction_single_supervised_epoch_" + str(epoch))


class TrainActionTypeDataPoint:
    """ Datapoint for training """

    def __init__(self, instruction, instruction_string, id_, gold_token_seq):
        self.instruction = instruction
        self.instruction_string = instruction_string
        self.task_id = id_
        self.gold_token_seq = gold_token_seq
