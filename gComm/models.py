import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.gumbel import Gumbel
from torch.distributions import Categorical
import numpy as np

# ========================Communication Channel============================ #
class CommChannel:
    """
    xyz
    """

    def __init__(self, msg_len, num_msgs, comm_type, temp, device):
        self.msg_len = msg_len
        self.num_msgs = num_msgs
        self.comm_type = comm_type
        self.temp = temp
        self.device = device

        self.gumbel = Gumbel(loc=torch.tensor([0.], device=self.device), scale=torch.tensor([1.], device=self.device))
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def binary(self, logits):
        """
        generates binary messages using logits

        :param: logits: [batch_size, msg_len, 2]
        """
        y = self.log_softmax(logits)
        y_hat = y + self.gumbel.sample(y.size()).squeeze(-1)
        y_hat = self.softmax(y_hat * self.temp)

        # Straight-Through Trick
        y_hard = torch.max(y_hat, dim=2)[1].to(self.device)
        y_soft = torch.zeros(y.size(0), y.size(1)).to(self.device)
        for i in range(y_soft.shape[0]):
            y_soft[i, :] = y_hat[i, :, :].gather(1, y_hard[i, :].view(-1, 1)).view(-1)
        ret = (y_hard.float() - y_soft).detach() + y_soft
        return ret

    def one_hot(self, probs, sample=False, dim=-1):
        """
        generates one-hot messages using sampling probabilities

        :param: sample: differentiable sample when True
        :param: probs: sampling probabilities [batch_size, msg_len]
        """
        cat_distr = RelaxedOneHotCategorical(self.temp, probs=probs)
        if sample:
            y_soft = cat_distr.sample()
        else:  # differentiable sample
            y_soft = cat_distr.rsample()

        # Straight-Through Trick
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft, device=self.device).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret


    def random(self):
        '''
        communicate random values
        '''
        message = torch.randn(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def fixed(self):
        '''
        communicate fixed values
        '''
        message = torch.ones(1, (self.num_msgs * self.msg_len), dtype=torch.float32).to(self.device)
        return message

    def perfect(self, concept_representation):
        '''
        @concept_representation :
        '''
        message = np.zeros(12, dtype=np.float32)
        message[:8] = concept_representation[4:12]  # size, color
        message[-4:] = concept_representation[-4:]  # task
        message = torch.tensor(message, dtype=torch.float32).unsqueeze(0).to(self.device)
        return message

    def oracle(self):
        '''
        we don't need this function
        '''
        pass

# ========================Speaker Module============================ #
class SpeakerBot(nn.Module):
    """
    Speaker has access to the 18-d concept input (size + shape + color + weight + task)
    and transmits 'num_msgs' messages of length 'msg_len'
    input_size: 18-d
    """

    def __init__(self, comm_type, input_size, hidden_size, output_size, num_msgs, temp, device):
        super().__init__()
        # model params
        self.comm_type = comm_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # msg_len
        self.num_msgs = num_msgs
        self.device = device

        self.rnn_cell = nn.RNNCell(self.input_size, self.hidden_size)

        if comm_type == 'categorical':
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        elif comm_type == 'binary':
            self.output_layer = nn.Linear(self.hidden_size, self.output_size*2)

        self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=self.device), requires_grad=True)
        self.comm_channel = CommChannel(msg_len=output_size, num_msgs=num_msgs,
                                        comm_type=comm_type, temp=temp, device=device)

    def forward(self, data_input, validation=False):
        """
        :param data_input: concept representation
        :param validation: if True, take argmax(.) over logits/probs
        """
        # model code
        batch_size = data_input.size(0)
        decoder_hidden = self.init_input
        message = []
        entropy = torch.zeros((batch_size,)).to(self.device)
        log_probs = torch.tensor(0.).to(self.device)

        for ind in range(self.num_msgs):
            # split_input = data_input[:, ind * 4: (ind + 1) * 4]
            decoder_hidden = self.rnn_cell(data_input, decoder_hidden)
            logits = self.output_layer(decoder_hidden)

            if self.comm_type == 'categorical':
                probs = F.softmax(logits, dim=-1)
                if not validation:
                    predict = self.comm_channel.one_hot(probs, sample=False)
                else:
                    predict = F.one_hot(torch.argmax(probs, dim=1), num_classes=self.output_size).float()

            elif self.comm_type == 'binary':
                logits = logits.view(batch_size, self.output_size, -1)
                binary_probs = torch.softmax(logits, dim=-1)
                probs = torch.zeros(batch_size, self.output_size).to(self.device)
                if not validation:
                    predict = self.comm_channel.binary(logits)
                else:
                    predict = torch.max(logits, dim=2)[1].to(self.device)
                for i in range(batch_size):
                    probs[i, :] = binary_probs[i, :, :].gather(1, predict[i, :].view(-1, 1)).view(-1)

            elif self.comm_type == 'continuous':
                probs = F.softmax(logits, dim=-1)
                predict = logits

            log_probs += torch.log((probs * predict).sum(dim=1)).squeeze(0)
            message.extend(predict)
            decoder_hidden = predict.detach()

        message = torch.stack(message)
        return message, log_probs, entropy


# =========================Target encoder Module============================= #
class TargetEncoder(nn.Module):
    """
    identifies the target in context of the distractors
    by using the received messages + grid input obtained
    """

    def __init__(self, grid_size, num_msgs, message_len):
        super().__init__()
        self.grid_size = grid_size
        self.project_target = nn.Linear(num_msgs * message_len, 17)  # [(num messages * message_length), grid_channels]

    def attention(self, grid_image, gru_projected):
        """
        obtain weights by taking the dot product between the target representation (using messages)
        and each cell of the grid.

        weights are added to each cell as an additional dimension
        """
        weights = torch.bmm(grid_image, gru_projected.view(-1, 17, 1))
        return weights

    def forward(self, speaker_out, grid_image):
        grid_image = grid_image
        gru_projected = self.project_target(speaker_out)
        attention = self.attention(grid_image, gru_projected)
        new_grid = torch.cat([grid_image, attention], dim=2).permute(0, 2, 1).contiguous().view(1, 18,
                                                                                                self.grid_size,
                                                                                                self.grid_size)
        return new_grid


# ==========================Grid Encoder Module========================= #
class GridEncoder(nn.Module):
    """
    processes the latent grid representation (i.e. the output from the target encoder)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(1, 1))

    def forward(self, grid_representation):
        batch_size = grid_representation.shape[0]
        state = self.conv(grid_representation).view(batch_size, -1)
        return state


# ============================Listener Module=============================== #
class ListenerBot(nn.Module):
    """
    Uses the state (processed grid information from the grid encoder) for sequential decision making
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1  # 150
        self.hidden_dim2 = hidden_dim2  # 30
        self.num_actions = num_actions

        self.actor_layer = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim1),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_dim1, self.hidden_dim2),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_dim2, self.num_actions))

    def forward(self, state, validate):
        policy_logits = self.actor_layer(state)
        # return policy_logits
        policy_dist = torch.softmax(policy_logits, dim=-1)

        if validate:
            action = torch.argmax(policy_dist, dim=1)
        else:
            action = Categorical(probs=policy_dist).sample()

        # action_freq[action_IND_to_STR(action.item())] += 1
        log_prob = Categorical(probs=policy_dist).log_prob(action)
        entropy = -(torch.log(policy_dist + 1e-8) * policy_dist).sum()
        # reward, done = environment_step(action=action.item())
        return log_prob, entropy, action_IND_to_STR(action.item())