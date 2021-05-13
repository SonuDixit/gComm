import numpy as np
from typing import List
from typing import Any
import matplotlib.pyplot as plt

np.random.seed(10)


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


def save_counter(description, counter, file):
    file.write(description + ": \n")
    for key, occurrence_count in counter.items():
        file.write("   {}: {}\n".format(key, occurrence_count))


def bar_plot(values: dict, title: str, save_path: str, y_axis_label="Occurrence"):
    sorted_values = list(values.items())
    sorted_values = [(y, x) for x, y in sorted_values]
    sorted_values.sort()
    values_per_label = [value[0] for value in sorted_values]

    sorted_errors = None

    labels = [value[1] for value in sorted_values]
    assert len(labels) == len(values_per_label)
    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values_per_label, yerr=sorted_errors, align='center', alpha=0.5)
    plt.gcf().subplots_adjust(bottom=0.2, )
    plt.xticks(y_pos, labels, rotation=90, fontsize="xx-small")
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.savefig(save_path)
    plt.close()


def grouped_bar_plot(values: dict, group_one_key: Any, group_two_key: Any, title: str, save_path: str,
                     y_axis_label="Occurence", sort_on_key=True):
    sorted_values = list(values.items())
    if sort_on_key:
        sorted_values.sort()
    values_group_one = [value[1][group_one_key] for value in sorted_values]
    values_group_two = [value[1][group_two_key] for value in sorted_values]

    sorted_errors_group_one = None
    sorted_errors_group_two = None

    labels = [value[0] for value in sorted_values]
    assert len(labels) == len(values_group_one)
    assert len(labels) == len(values_group_two)
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    width = 0.35
    p1 = ax.bar(y_pos, values_group_one, width, yerr=sorted_errors_group_one, align='center', alpha=0.5)
    p2 = ax.bar(y_pos + width, values_group_two, width, yerr=sorted_errors_group_two, align='center', alpha=0.5)
    plt.gcf().subplots_adjust(bottom=0.2, )
    plt.xticks(y_pos, labels, rotation=90, fontsize="xx-small")
    plt.ylabel(y_axis_label)
    plt.title(title)
    ax.legend((p1[0], p2[0]), (group_one_key, group_two_key))

    plt.savefig(save_path)
    plt.close()


def generate_task_progress(task_reward_dict, color, interval, file_name=None):
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


def plot_d_loss_progress(net_d_loss, file_name=None):
    fig = plt.figure()
    x = np.arange(len(net_d_loss))
    plt.plot(x, net_d_loss, label='d loss')
    plt.legend()
    plt.title('Discriminator Loss (Train)')
    if file_name is None:
        plt.savefig('d_loss_progress.png')
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


def generate_distractors(actual_encoding):
    shape, color = actual_encoding[:3], actual_encoding[3:7]
    # size_bit = np.random.choice([i for i in range(len(size)) if i != np.array(size).argmax()])
    shape_bit = np.random.choice([i for i in range(len(shape)) if i != np.array(shape).argmax()])
    color_bit = np.random.choice([i for i in range(len(color)) if i != np.array(color).argmax()])
    # weight_bit = np.random.choice([i for i in range(len(weight)) if i != np.array(weight).argmax()])

    # new_size = [0 if i != size_bit else 1 for i in range(len(size))]
    new_shape = [0 if i != shape_bit else 1 for i in range(len(shape))]
    new_color = [0 if i != color_bit else 1 for i in range(len(color))]
    # new_weight = [0 if i != weight_bit else 1 for i in range(len(weight))]

    # distractors = [shape + color + weight]
    distractors = [new_shape + color]
    distractors += [shape + new_color]

    true_index = np.random.randint(3)
    distractors.insert(true_index, actual_encoding)

    return distractors, true_index


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


def adaptive_rate(parameter, iteration, type):
    if type == 'exponential':
        parameter = parameter * (0.1 ** (iteration // 500))
    elif type == 'linear':
        parameter = max(parameter - 1e-5 * iteration, 0.01)
    return parameter


def action_IND_to_STR(action: int):
    actions_dict = {'left': 0, 'right': 1, 'forward': 2, 'backward': 3,
                    'push': 4, 'pull': 5, 'pickup': 6, 'drop': 7}
    inv_actions_dict = {v: k for k, v in actions_dict.items()}
    return inv_actions_dict[action]
