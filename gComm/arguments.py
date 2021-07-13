import argparse


def Arguments():
    parser = argparse.ArgumentParser(description="gComm")

    # ------------------------------- General arguments ------------------------------- #

    parser.add_argument('--output_directory', type=str, default='save_folder', help='save episodes')

    # ------------------------------- Environment arguments ------------------------------- #

    parser.add_argument('--grid_size', type=int, default=4, help='Number of rows (and columns) in the grid world. '
                                                                 'Note: only odd-sized grids possible with enable '
                                                                 'maze feature')
    parser.add_argument('--min_other_objects', type=int, default=4,
                        help='Minimum amount of objects to put in the grid world.')
    parser.add_argument('--max_objects', type=int, default=4, help='Maximum number of OTHER objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--min_object_size', type=int, default=1, help='Smallest object size.')
    parser.add_argument('--max_object_size', type=int, default=4, help='Biggest object size.')
    parser.add_argument('--other_objects_sample_percentage', type=float, default=0.3,
                        help='Percentage of possible objects distinct from the target to place in the world. '
                             'lies in the range [0,1]')

    # ------------------------------------ Maze parameters ---------------------------------------- #

    parser.add_argument('--obstacles_flag', action='store_true', default=False,
                        help='whether to add obstacles to the gridworld (block obstacles)')
    parser.add_argument('--num_obstacles', type=int, default=5, help='Count of obstacles in the gridworld')
    parser.add_argument('--enable_maze', action='store_true', default=False, help='enable maze-grid (default: False)'
                                                                                  'Note: only odd-sized grids possible '
                                                                                  'with enable maze feature')
    parser.add_argument('--maze_complexity', type=float, default=0, help='maze complexity lies in the range [0,1]')
    parser.add_argument('--maze_density', type=float, default=0, help='maze density lies in the range [0,1]')

    # ------------------------------- Grammar and Vocabulary arguments ------------------------------- #

    parser.add_argument('--type_grammar', type=str, default='simple_intrans', required=False,
                        choices=['simple_intrans', 'simple_trans', 'normal'])
    parser.add_argument('--intransitive_verbs', type=str, default='walk',
                        help='Comma-separated list of intransitive verbs.')
    parser.add_argument('--transitive_verbs', type=str, default='push,pull,pickup,drop',
                        help='Comma-separated list of transitive verbs. (also possible: push,pull,pickup,drop)')
    parser.add_argument('--nouns', type=str, default='circle,square,cylinder,diamond', required=False,
                        help='Comma-separated list of nouns.')
    parser.add_argument('--color_adjectives', type=str, default='red,blue,yellow,green', required=False,
                        help='Comma-separated list of colors.')
    parser.add_argument('--size_adjectives', type=str, default='', required=False,
                        help='Comma-separated list of sizes.')
    parser.add_argument('--keep_fixed_weights', action='store_true', default=True,
                        help='if True, then weights are fixed such that sizes 1,2 are light and sizes 3,4 are heavy; '
                             'otherwise, weights are randomly fixed for different sizes')
    parser.add_argument('--all_light', action='store_true', default=False, help='make all objects light (weight=1)')

    # ------------------------------------ RL-framework ----------------------------------------- #

    parser.add_argument('--num_episodes', type=int, default=400000, help='number of episodes', required=False)
    parser.add_argument('--episode_len', type=int, default=10, help='length of episode', required=False)
    parser.add_argument('--grid_input_type', type=str, default='vector',
                        choices=['image', 'vector', 'with_target'], required=False,
                        help='observed grid input (image: image input of the grid view; '
                             'vector: vector representation of each cell,'
                             'with_target: vector representation with target specified)')

    # ------------------------------------ Communication Channel ----------------------------------------- #
    parser.add_argument('--comm_type', type=str, default='categorical',
                        choices=['continuous', 'binary', 'categorical', 'random', 'fixed', 'perfect', 'oracle'])
    parser.add_argument('--comm_setting', type=str, default='cheap_talk',
                        choices=['cheap_talk', 'costly_signalling'], help='TODO: costly signalling')
    parser.add_argument('--temp', type=float, default=1, help='temperature parameter for discrete messages')

    # ------------------------------------ lights off feature ----------------------------------------- #
    parser.add_argument('--lights_out', action='store_true', default=False, help='Darker grid when lights out is True')

    # ----------------------------------- visualisation -----------------------------------------------
    parser.add_argument('--render_episode', action='store_true', default=False, help='whether to render every step')
    parser.add_argument('--wait_time', type=float, default=0.3, help='wait time between consecutive time-steps')

    return vars(parser.parse_args())


def pass_arguments():
    flags = Arguments()
    return flags
