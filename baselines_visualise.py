"""
baselines:
        Random, Fixed, Perfect, Oracle, Simple (Categorical)

Actions:
        left = 0
        right = 1
        forward = 2
        backward = 3
        push = 4
        pull = 5
        pickup = 6
        drop = 7
"""

from gComm.arguments import Arguments
from gComm.gComm_env import gCommEnv
from baseline_models import SpeakerBot, ListenerBot
from gComm.agent import SpeakerAgent, ListenerAgent

import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flags = Arguments()

GAMMA = 0.9  # discount rate
LAMBDA = 0.01  # hyper parameter for entropy
num_msgs = 3  # n_m of the communication channel
# d_m of the communication channel
if flags['comm_type'] == 'binary':
    msg_len = 2  # (binary)
else:
    msg_len = 4  # (categorical or others)

# number of actions by Listener-Bot
num_actions = 4
if flags['type_grammar'] == 'simple_trans':
    num_actions += len(flags['transitive_verbs'].split(','))

# ======================= Data storage paths ========================== #
run_id = 1
path = os.path.join(os.getcwd(), flags['comm_type'] + "_speaker_data/") + "run_" + str(run_id) + "/"
model_load_path = os.path.join(path , "checkpoint_dir")

visual_path = os.path.join(path, 'visual') # string name of folder : 'test_time'
SAVE_FIG = True  # save test episodes?
LOAD_AND_VISUAL = False
speaker_weight_itr = 76000
listener_weight_itr = 76000

if SAVE_FIG:
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

# =======================Save Models======================== #
def save_models(iteration):
    if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
        all_vars['speaker_agent'].save(model_load_path, iteration)
    all_vars['listener_agent'].save(model_load_path, iteration)


def single_loop(env, validation=True, episode_save_path=None):
    instruction, verb_in_command = env.generate_world(
        other_objects_sample_percentage=flags['other_objects_sample_percentage'],
        max_other_objects=flags['max_objects'],
        min_other_objects=flags['min_other_objects'],
        num_obstacles=flags['num_obstacles'])

    # concept input (encoded instruction and target information)
    concept_representation, weight = env.concept_input(verb_in_command)

    actions = ''
    log_probs = []
    rewards = []
    entropy_term = torch.tensor(0.).to(DEVICE)

    # speaker model processes instruction based on baseline specification
    speaker_out = speaker_agent.transmit(concept=concept_representation, validation=validation)

    # time-steps
    for t in range(flags["episode_len"]):
        # grid input
        if flags["grid_input_type"] == "image":
            grid_representation = env.grid_image_input()  # [img_height, img_width, 3]
        elif flags["grid_input_type"] == "vector":
            grid_vector_size = 17
            grid_representation = env.grid_input()  # [grid_height, grid_width, num_channels]
            grid_representation = torch.tensor(grid_representation,
                                               dtype=torch.float32).contiguous().view(1,
                                                                                      flags["grid_size"] ** 2,
                                                                                      grid_vector_size).to(DEVICE)
        # baseline: ORACLE LISTENER
        elif flags["grid_input_type"] == "with_target":
            grid_representation = \
                env.grid_input(specify_target=True)  # [grid_height, grid_width, num_channels+1]
            grid_vector_size = 18
            grid_representation = torch.tensor(grid_representation,
                                               dtype=torch.float32).contiguous().view(1,
                                                                                      flags["grid_size"] ** 2,
                                                                                      grid_vector_size).to(DEVICE)

        # render each step of the episode
        # if flags['render_episode']:
        env.render_episode(mission=instruction,
                           countdown=(flags["episode_len"] - t),
                           actions=actions,
                           weight=weight,
                           verb_in_command=verb_in_command,
                           save_path=episode_save_path,
                           save_fig=True)

        # action by policy
        log_prob, entropy, action = listener_agent.act(state=(grid_representation, speaker_out), validate=validation)

        # reward at each time-step;
        # done flag indicates whether the task was completed
        reward, done = env.step(action)
        actions += action + ' '

        rewards.append(reward)
        log_probs.append(log_prob.squeeze(0))
        entropy_term += entropy

        if done:
            # render each step of the episode
            # if flags['render_episode']:
            env.render_episode(mission=instruction,
                               countdown=(flags["episode_len"] - t - 1),
                               actions=actions,
                               weight=weight,
                               verb_in_command=verb_in_command,
                               save_path=episode_save_path,
                               save_fig=True)
            val = torch.tensor(0., device=DEVICE)
            break

    if validation:
        return np.array(rewards).sum()

    train_rewards = torch.tensor(rewards).to(DEVICE)
    vals = []
    for t in reversed(range(len(rewards))):
        val = train_rewards[t] + GAMMA * val
        vals.insert(0, val)

    vals = torch.stack(vals)
    log_probs = torch.stack(log_probs)
    advantage = vals.detach()

    agent_loss = (-log_probs * advantage.detach()).mean()
    net_loss = agent_loss - LAMBDA * entropy_term
    return net_loss, train_rewards


def main():

    # initializing the vocabulary
    intransitive_verbs = flags["intransitive_verbs"].split(',')
    transitive_verbs = flags["transitive_verbs"].split(',')
    nouns = flags["nouns"].split(',')
    color_adjectives = flags["color_adjectives"].split(',') if flags["color_adjectives"] else []
    size_adjectives = flags["size_adjectives"].split(',') if flags["size_adjectives"] else []

    # initializing the environment
    env = gCommEnv(
        intransitive_verbs=intransitive_verbs, transitive_verbs=transitive_verbs, nouns=nouns,
        color_adjectives=color_adjectives, size_adjectives=size_adjectives,
        min_object_size=flags["min_object_size"], max_object_size=flags["max_object_size"],
        save_directory=flags["output_directory"], grid_size=flags["grid_size"],
        type_grammar=flags["type_grammar"], maze_complexity=flags["maze_complexity"],
        maze_density=flags["maze_density"], enable_maze=flags["enable_maze"],
        lights_out=flags["lights_out"], obstacles_flag=flags['obstacles_flag'],
        keep_fixed_weights=flags["keep_fixed_weights"], all_light=flags['all_light'],
        episode_len=flags["episode_len"], wait=flags['wait_time'])


    test_episodes = 10
    val_rewards = 0
    for Episode_count in range(1, test_episodes + 1):
        if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
            all_vars['speaker_agent'].eval()
        all_vars['listener_agent'].eval()
        episode_save_path = os.path.join(visual_path,'example_'+str(Episode_count))
        if not os.path.exists(episode_save_path):
            os.makedirs(episode_save_path)
        with torch.no_grad():
            val_reward = single_loop(env, validation=True, episode_save_path=episode_save_path)
            val_rewards += val_reward

    print('val-reward: {}'.format(val_rewards/test_episodes))

if __name__ == "__main__":
    # =================initialize model params and environment============== #

    # ================== Listener-Bot ====================== #
    oracle = True if flags['comm_type'] == 'oracle' else False
    listener_bot = ListenerBot(grid_size=flags['grid_size'], num_msgs=num_msgs, msg_len=msg_len,
                               in_channels=18, out_channels=20, input_dim=320, hidden_dim1=150,
                               hidden_dim2=30, num_actions=num_actions, oracle=oracle).to(DEVICE)
    listener_agent = ListenerAgent(listener_model=listener_bot)
    listener_agent.load(model_weight_path=model_load_path, iteration=listener_weight_itr)

    # ================== Speaker-Bot ====================== #
    speaker_bot = SpeakerBot(comm_type=flags['comm_type'], input_size=12, hidden_size=4,
                             output_size=msg_len, num_msgs=num_msgs, device=DEVICE).to(DEVICE)
    speaker_agent = SpeakerAgent(num_msgs=num_msgs, msg_len=msg_len, comm_type=flags['comm_type'],
                                 temp=flags['temp'], speaker_model=speaker_bot, device=DEVICE)

    all_params = list(listener_agent.listener_model.parameters())
    if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
        all_params += list(speaker_agent.speaker_model.parameters())
        speaker_agent.load(model_weight_path=model_load_path, iteration = speaker_weight_itr)
        # load weights

    all_vars = {'listener_agent': listener_agent, 'speaker_agent': speaker_agent}



    print('\n=================== Saving: {} ===================\n'.format(flags['comm_type'].upper()))
    main()

