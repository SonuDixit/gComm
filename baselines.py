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
from gComm.helpers import generate_task_progress

import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

SAVE_FIG = False  # save test episodes?
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
model_load_path = Path(path + "checkpoint_dir")
plot_path = Path(path + "plots")
model_load_path.mkdir(exist_ok=True, parents=True)
plot_path.mkdir(exist_ok=True, parents=True)
log_file = open(path + "log.txt", "w")
log_file.write(str(flags) + '\n')
visualization_path = None
save_model_flag = False


# =======================Save Models======================== #
def save_models(iteration):
    for model_name, model in all_vars.items():
        if model_name in ['target_encoder', 'grid_encoder', 'listener_bot']:
            with open(os.path.join(model_load_path, str(iteration) + '_' + model_name), 'wb') as f:
                torch.save(model.state_dict(), f)


def single_loop(env, validation=False):
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
        if flags['render_episode']:
            env.render_episode(mission=instruction,
                               countdown=(flags["episode_len"] - t),
                               actions=actions,
                               weight=weight,
                               verb_in_command=verb_in_command,
                               save_path=visualization_path,
                               save_fig=False)

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
            if flags['render_episode']:
                env.render_episode(mission=instruction,
                                   countdown=(flags["episode_len"] - t - 1),
                                   actions=actions,
                                   weight=weight,
                                   verb_in_command=verb_in_command,
                                   save_path=visualization_path,
                                   save_fig=SAVE_FIG)
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
    # Create directory for visualizations if it doesn't exist.
    flags['output_directory'] = os.path.join(os.getcwd(), flags['output_directory'])
    if flags['output_directory']:
        visualization_path = flags['output_directory']
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

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

    # generating episodes
    task_rewards = {"'walk'": []}
    for Episode_count in range(1, flags["num_episodes"] + 1):
        # ==================== Train =================== #
        if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
            all_vars['speaker_bot'].train(True)
        all_vars['listener_bot'].train(True)
        all_vars['optimizer'].zero_grad()
        # run a single training loop
        net_loss, train_rewards = single_loop(env, validation=False)
        net_loss.backward()
        all_vars['optimizer'].step()

        # ==================== validation ====================== #
        if Episode_count % 500 == 0:
            val_rewards = 0
            num_val_iter = 30  # number of validation loops/episodes to be run
            if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
                all_vars['speaker_bot'].eval()
            all_vars['listener_bot'].eval()
            with torch.no_grad():
                for _ in range(num_val_iter):
                    val_reward = single_loop(env, validation=True)
                    val_rewards += val_reward / num_val_iter
            print('episode: {} | val-reward: {}'.format(Episode_count, val_rewards))
            task_rewards["'walk'"].extend([val_rewards])

        # ========================== Plot ============================= #
        if Episode_count % 2000 == 0 and Episode_count != 0:
            print('------------------Generating plots------------------------')
            generate_task_progress(task_reward_dict=task_rewards, color='m',
                                   file_name=os.path.join(plot_path, 'task_progress.png'))

        # =====================save model====================== #
        if save_model_flag is True:
            if Episode_count % 2000 == 0:
                print('------------------Saving model checkpoint------------------\n')
                save_models(iteration=Episode_count)

        log_file.write('Episode: ' + str(Episode_count) + ' | Train Reward: ' + str(train_rewards) +
                       ' | Train Loss: ' + str(net_loss) + "\n")

        log_file.flush()


if __name__ == "__main__":
    # =================initialize model params and environment============== #

    # ================== Listener-Bot ====================== #
    oracle = True if flags['comm_type'] == 'oracle' else False
    listener_bot = ListenerBot(grid_size=flags['grid_size'], num_msgs=num_msgs, msg_len=msg_len,
                               in_channels=18, out_channels=20, input_dim=320, hidden_dim1=150,
                               hidden_dim2=30, num_actions=num_actions, oracle=oracle).to(DEVICE)
    listener_agent = ListenerAgent(listener_model=listener_bot)

    # ================== Speaker-Bot ====================== #
    speaker_bot = SpeakerBot(comm_type=flags['comm_type'], input_size=12, hidden_size=4,
                             output_size=msg_len, num_msgs=num_msgs, device=DEVICE).to(DEVICE)
    speaker_agent = SpeakerAgent(num_msgs=num_msgs, msg_len=msg_len, comm_type=flags['comm_type'],
                                 temp=flags['temp'], speaker_model=speaker_bot, device=DEVICE)

    all_params = list(listener_agent.listener_model.parameters())
    if flags['comm_type'] in ['categorical', 'continuous', 'binary']:
        all_params += list(speaker_agent.speaker_model.parameters())

    optimizer = optim.Adam(all_params, lr=4e-4)
    all_vars = {'listener_bot': listener_bot, 'speaker_bot': speaker_bot, 'optimizer': optimizer}

    print('\n=================== Baseline: {} ===================\n'.format(flags['comm_type'].upper()))
    main()

    log_file.close()
