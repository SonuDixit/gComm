import numpy as np
import pandas as pd
from gComm.helpers import display_table


def hamming_dist(concept1, concept2):
    """
        Calculate the hamming distance of two concepts.
        The input concepts should be tuple, e.g. ('red','box')
        We require concept1 and 2 have the same number of attributes,
        i.e., len(concept1)==len(concept2)
    """
    acc_dist = 0
    for i in range(len(concept1)):
        if concept1[i] != concept2[i]:
            acc_dist += 1

    return acc_dist


def edit_dist(str1, str2):
    """
        Calculate the edit distance of two strings.
        Insert/delete/replace all cause 1.
    """
    len1, len2 = len(str1), len(str2)
    DM = [0]
    for i in range(len1):
        DM.append(i + 1)

    for j in range(len2):
        DM_new = [j + 1]
        for i in range(len1):
            tmp = 0 if str1[i] == str2[j] else 1
            new = min(DM[i + 1] + 1, DM_new[i] + 1, DM[i] + tmp)
            DM_new.append(new)
        DM = DM_new

    return DM[-1]


def topsim_metric(msg):
    """
        Calculate the compositionalities using metric mentioned in:
        Language as an evolutionary system -- Appendix A (Kirby 2005)
        Input: dictionary
            msg = {('0', '0'):'aa', ('0', '1'):'bb', ('1', '0'):'ab', ('1', '1'):'ba'}
            keys = concept symbols ('<shape>', '<color>')
            values = message symbols
        Output:
            corr_pearson:   person correlation
            corr_spearman:  spearman correlation
    """
    keys_list = list(msg.keys())
    concept_pairs = []
    message_pairs = []
    # ===== Form concepts and message pairs ========
    for i in range(len(keys_list)):
        # for j in range(i+1, len(keys_list)):
        for j in range(len(keys_list)):
            tmp1 = (keys_list[i], keys_list[j])
            concept_pairs.append((keys_list[i], keys_list[j]))
            tmp2 = (msg[tmp1[0]], msg[tmp1[1]])
            message_pairs.append(tmp2)

    # ===== Calculate distant for these pairs ======
    concept_HD = []
    message_ED = []
    for i in range(len(concept_pairs)):
        concept1, concept2 = concept_pairs[i]
        message1, message2 = message_pairs[i]
        concept_HD.append(hamming_dist(concept1, concept2))
        message_ED.append(edit_dist(message1, message2))

    if np.sum(message_ED) == 0:
        message_ED = np.asarray(message_ED) + 0.1
        message_ED[-1] -= 0.01

    dist_table = pd.DataFrame({'HD': np.asarray(concept_HD),
                               'ED': np.asarray(message_ED)})
    corr_pearson = dist_table.corr()['ED']['HD']
    corr_spearman = dist_table.corr('spearman')['ED']['HD']

    return corr_pearson, corr_spearman


if __name__ == "__main__":
    print('# ================== Demos ================== #')
    # Demo 1: perfectly compositional
    messages = {('green', 'box'): 'aa', ('blue', 'box'): 'ba', ('green', 'circle'): 'ab', ('blue', 'circle'): 'bb'}
    c_p, c_s = topsim_metric(msg=messages)
    display_table(messages=messages, protocol='perfectly compositional', corr=(c_p, c_s))

    # Demo 2: two different objects (concepts) map to the same set of messages
    messages = {('green', 'box'): 'ab', ('blue', 'box'): 'ba', ('green', 'circle'): 'ab', ('blue', 'circle'): 'bb'}
    c_p, c_s = topsim_metric(msg=messages)
    display_table(messages=messages, protocol='surjective (not injective)', corr=(c_p, c_s))

    # Demo 3: holistic language (one-to-one mapping but not fully systematic)
    messages = {('green', 'box'): 'ba', ('blue', 'box'): 'aa', ('green', 'circle'): 'ab', ('blue', 'circle'): 'bb'}
    c_p, c_s = topsim_metric(msg=messages)
    display_table(messages=messages, protocol='holistic', corr=(c_p, c_s))

    # Demo 3: ambiguous language
    messages = {('green', 'box'): 'aa', ('blue', 'box'): 'aa', ('green', 'circle'): 'aa', ('blue', 'circle'): 'aa'}
    c_p, c_s = topsim_metric(msg=messages)
    display_table(messages=messages, protocol='ambiguous language', corr=(c_p, c_s))
