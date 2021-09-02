import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================Speaker Module============================ #
class SpeakerBot(nn.Module):
    """
    Speaker has access to the 18-d concept input (size + shape + color + weight + task)
    and transmits 'num_msgs' (n_m) messages of length 'msg_len' (d_m)
    """

    def __init__(self, comm_type, input_size, hidden_size, output_size, num_msgs, device):
        """
        :comm_type: ['categorical', 'binary', 'continuous']
        :input size: 18-d (concept representation)
        :hidden_size: hidden size of RNN
        :output_size: msg_len (d_m)
        :num_msgs: num_msgs (n_m)
        :device: cpu/gpu
        """
        super().__init__()
        # model params
        self.comm_type = comm_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # msg_len
        self.num_msgs = num_msgs
        self.device = device

        self.rnn_cell = nn.RNNCell(self.input_size, self.hidden_size)

        if comm_type == 'categorical' or comm_type == 'continuous':
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        elif comm_type == 'binary':
            self.output_layer = nn.Linear(self.hidden_size, self.output_size * 2)

        self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=self.device), requires_grad=True)

    def forward(self, data_input, comm_channel, validation=False):
        """
        :param data_input: concept representation
        :param comm_channel: object of class CommChannel in agent.py
        :param validation: if True, take argmax(.) over logits/probs
        """
        # model code
        batch_size = data_input.size(0)
        decoder_hidden = self.init_input
        message = []
        entropy = torch.zeros((batch_size,)).to(self.device)
        log_probs = torch.tensor(0.).to(self.device)

        for ind in range(self.num_msgs):
            decoder_hidden = self.rnn_cell(data_input, decoder_hidden)
            logits = self.output_layer(decoder_hidden)

            if self.comm_type == 'categorical':
                probs = F.softmax(logits, dim=-1)
                if not validation:
                    predict, _ = comm_channel.one_hot(probs)
                else:
                    predict = F.one_hot(torch.argmax(probs, dim=1), num_classes=self.output_size).float()

            elif self.comm_type == 'binary':
                logits = logits.view(batch_size, self.output_size, -1)
                binary_probs = torch.softmax(logits, dim=-1)
                probs = torch.zeros(batch_size, self.output_size).to(self.device)
                if not validation:
                    predict, probs = comm_channel.binary(logits)
                else:
                    predict = torch.max(logits, dim=2)[1].type(torch.float32).to(self.device)
                for i in range(batch_size):
                    probs[i] = binary_probs[i].gather(1, predict[i].view(-1, 1).dtype(torch.int64)).view(-1)

            elif self.comm_type == 'continuous':
                probs = F.softmax(logits, dim=-1)
                predict = comm_channel.continuous(logits)

            log_probs += torch.log((probs * predict).sum(dim=1)).squeeze(0)
            message.extend(predict)
            decoder_hidden = predict.detach()

        message = torch.stack(message)
        # return message, log_probs, entropy
        return message


class RandomSpeaker:
    def __init__(self, comm_type, num_msgs, msg_len):
        pass


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

    @staticmethod
    def attention(grid_image, gru_projected):
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
        attention = TargetEncoder.attention(grid_image, gru_projected)
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
    > Steps:
    1. Target Encoder processes the received messages and the current grid input
    and generates a new grid representation containing the target and task specifics
    2. Grid Encoder uses the new grid representation and convolves over each cell using 1*1 conv;
    the output of the Grid Encoder is the state
    3. Listener Action Layer uses the state to take actions
    """

    def __init__(self, grid_size, num_msgs, msg_len, in_channels, out_channels,
                 input_dim, hidden_dim1, hidden_dim2, num_actions, oracle=False):
        """
        > Target Encoder
        :grid_size: 4 * 4 for baselines
        :num_msgs: n_m
        :msg_len: d_m
        :oracle: implement Oracle Listener if True (sidestep Target Encoder)

        > Grid Encoder
        :in_channels: input channels of the grid encoder (input dim of processed grid representation)
        :out_channels: out channels of the grid encoder

        > Action layer
        :input_dim: dim of flattened output of the grid encoder
        :hidden_dim1: dim of the latent representation
        :hidden_dim2: dim of the latent representation
        :num_actions: size of action space (WALK task: 4, PUSH & PULL: 6)
        """
        super().__init__()
        if not oracle:
            self.target_encoder = TargetEncoder(grid_size=grid_size, num_msgs=num_msgs, message_len=msg_len)
        self.grid_encoder = GridEncoder(in_channels=in_channels, out_channels=out_channels)
        self.oracle = oracle

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_actions = num_actions

        self.action_layer = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim1),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim1, self.hidden_dim2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim2, self.num_actions))

    def forward(self, grid_image, speaker_out=None):
        if not self.oracle:
            grid_image = self.target_encoder(speaker_out, grid_image)
        else:
            grid_image = grid_image.contiguous().permute(2, 0, 1).unsqueeze(0)
        state = self.grid_encoder(grid_image)
        policy_logits = self.action_layer(state)
        return policy_logits
