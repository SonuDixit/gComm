import numpy as np
from typing import List
import matplotlib.pyplot as plt

# np.random.seed(10)


def topo_sort(items, constraints):
    if not constraints:
        return items
    items = list(items)
    constraints = list(constraints)
    out = []
    while len(items) > 0:
        roots = [
            i for i in items
            if not any(c[1] == i for c in constraints)
        ]
        assert len(roots) > 0, (items, constraints)
        to_pop = roots[0]
        items.remove(to_pop)
        constraints = [c for c in constraints if c[0] != to_pop]
        out.append(to_pop)
    return out


def one_hot(size: int, idx: int) -> np.ndarray:
    one_hot_vector = np.zeros(size, dtype=int)
    one_hot_vector[idx] = 1
    return one_hot_vector


def generate_possible_object_names(color: str, shape: str) -> List[str]:
    # TODO: does this still make sense when size is not small or large
    names = [shape, ' '.join([color, shape])]
    return names


def generate_task_progress(task_reward_dict, color, file_name=None):
    fig = plt.figure()
    for task, rewards in task_reward_dict.items():
        acl_iter = np.arange(len(rewards))
        plt.plot(acl_iter, rewards, label=task, color=color)
    plt.legend()
    plt.title('Task Rewards (Validation)')
    if file_name is None:
        plt.savefig('tasks_progress.png')
    else:
        plt.savefig(file_name)
    plt.close(fig)


def generate_task_frequency(task_freq_dict, file_name=None):
    fig = plt.figure()
    tasks = list(task_freq_dict.keys())
    freq = list(task_freq_dict.values())
    plt.bar(tasks, freq)
    plt.title('Task Frequency')
    if file_name is None:
        plt.savefig('tasks_frequency.png')
    else:
        plt.savefig(file_name)
    plt.close(fig)


def generate_actions_frequency(action_freq_dict, file_name=None):
    fig = plt.figure()
    actions = list(action_freq_dict.keys())
    freq = list(action_freq_dict.values())
    plt.bar(actions, freq, color='g')
    plt.title('Action Frequency')
    if file_name is None:
        plt.savefig('actions_frequency.png')
    else:
        plt.savefig(file_name)
    plt.close(fig)


def plot_topsim(pearson, spearman, file_name=None):
    fig = plt.figure()
    x = np.arange(len(pearson))
    plt.plot(x, pearson, label='pearson')
    plt.plot(x, spearman, label='spearman')
    plt.legend()
    plt.title('topsim')
    if file_name is None:
        plt.savefig('topsim.png')
    else:
        plt.savefig(file_name)
    plt.close(fig)


def binary2dec(msg_seq):
    """
    receives a seq of messages [seq_len, msg_dim].
    Processes the bits (of each message )into a string.
    Converts the string from binary to decimal.
    Concatenates the decimal output of all messages (of the seq)
    """
    concat = ''
    for msg in msg_seq:
        string = ''.join([str(int(i)) for i in msg])
        concat += str(int(string, 2))
    return concat


def action_IND_to_STR(action: int):
    actions_dict = {'left': 0, 'right': 1, 'forward': 2, 'backward': 3,
                    'push': 4, 'pull': 5, 'pickup': 6, 'drop': 7}
    inv_actions_dict = {v: k for k, v in actions_dict.items()}
    return inv_actions_dict[action]


def display_table(messages: dict, protocol: str, corr: tuple):
    print('\n ============ protocol: {} ============='.format(protocol))
    print("{:<15} {:<10}".format('Concept', 'Messages'))
    for k, v in messages.items():
        print("{:<15} {:<10}".format(' '.join(k), v))
    print("c_p = {} , c_s = {}".format(corr[0], corr[1]))
