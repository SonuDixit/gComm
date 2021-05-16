"""
user generated input actions to test package
"""

from gComm.arguments import Arguments
from gComm.gComm_env import gCommEnv

import os

RENDER = True  # render test episodes?
SAVE_FIG = False  # save test episodes?
# random.seed(104)
# np.random.seed(104)


def main():
    flags = Arguments()

    # Create directory for visualizations if it doesn't exist.
    visualization_path = None
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
        keep_fixed_weights=flags["keep_fixed_weights"], episode_len=flags["episode_len"])

    # generating episodes
    EP_count = 0
    for _ in range(flags["num_episodes"]):

        instruction, verb_in_command = env.generate_world(
            other_objects_sample_percentage=flags['other_objects_sample_percentage'],
            max_other_objects=flags['max_objects'],
            min_other_objects=flags['min_other_objects'],
            num_obstacles=flags['num_obstacles'])

        actions = ''
        # RENDER = True
        EP_count += 1
        print(EP_count)
        if RENDER:
            # time-steps
            for t in range(flags["episode_len"]):

                # grid input
                if flags["grid_input_type"] == "image":
                    _ = env.grid_image_input()  # [img_height, img_width, 3]
                elif flags["grid_input_type"] == "vector":
                    _ = env.grid_input()  # [grid_height, grid_width, num_channels]
                # baseline: ORACLE LISTENER
                elif flags["grid_input_type"] == "with_target":
                    _ = env.grid_input(specify_target=True)  # [grid_height, grid_width, num_channels+1]

                # concept input (encoded instruction and target information)
                _, weight = env.concept_input(verb_in_command)

                # render each step of the episode
                env.render_episode(mission=instruction,
                                   countdown=(flags["episode_len"] - t),
                                   actions=actions,
                                   weight=weight,
                                   verb_in_command=verb_in_command,
                                   save_path=visualization_path,
                                   save_fig=False)

                # input user action
                # modify this line to get output from policy module
                action = input('next action:')

                # reward at each time-step;
                # done flag indicates whether the task was completed
                reward, done = env.step(action)
                print('reward:', reward)
                actions += action + ' '

                if done:
                    # render each step of the episode
                    env.render_episode(mission=instruction,
                                       countdown=(flags["episode_len"] - t - 1),
                                       actions=actions,
                                       weight=weight,
                                       verb_in_command=verb_in_command,
                                       save_path=visualization_path,
                                       save_fig=SAVE_FIG)
                    break


if __name__ == "__main__":
    main()
