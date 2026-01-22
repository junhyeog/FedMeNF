# data split from pFL-Bench
# https://github.com/alibaba/FederatedScope
# https://github.com/shaoxiongji/federated-learning/blob/master/utils/sampling.py

import numpy as np


def iid(dataset, num_clients):
    all_idxs = [i for i in range(len(dataset))]
    all_idxs = np.random.permutation(all_idxs)
    all_idxs = np.array_split(all_idxs, num_clients)
    dict_clients = {i: all_idxs[i] for i in range(num_clients)}
    return dict_clients


def _split_according_to_prior(label, client_num, prior, min_size=1):
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]
        idx_k = np.random.permutation(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] * len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))]

    # make sure each client has at least min_size
    for i in range(len(idx_slice)):
        while len(idx_slice[i]) < min_size:
            random_client = np.random.choice(range(client_num))
            if len(idx_slice[random_client]) > min_size:
                idx_slice[i].append(idx_slice[random_client].pop())
            else:
                continue

    for i in range(len(idx_slice)):
        idx_slice[i] = np.random.permutation(idx_slice[i])

    dict_clients = {client_idx: np.array(idx_slice[client_idx]) for client_idx in range(client_num)}
    return dict_clients


def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=1, prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py
    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError("Only support single-label tasks!")

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior, min_size)

    num = len(label)
    classes = len(np.unique(label))
    assert num >= client_num * min_size, f"The number of sample should be " f"greater than" f" {client_num * min_size}."

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        # for label k
        idx_k = np.where(label == k)[0]
        idx_k = np.random.permutation(idx_k)
        prop = np.random.dirichlet(np.repeat(alpha, client_num))
        prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
        idx_k_slice = [idx.tolist() for idx in np.split(idx_k, prop)]
        idxs = np.arange(len(idx_k_slice))
        np.random.shuffle(idxs)
        idx_slice = [idx_j + idx_k_slice[idx] for idx_j, idx in zip(idx_slice, idxs)]

    # make sure each client has at least min_size
    for i in range(len(idx_slice)):
        while len(idx_slice[i]) < min_size:
            random_client = np.random.choice(range(client_num))
            if len(idx_slice[random_client]) > min_size:
                idx_slice[i].append(idx_slice[random_client].pop())
            else:
                continue

    for i in range(client_num):
        idx_slice[i] = np.random.permutation(idx_slice[i])

    dict_clients = {client_idx: np.array(idx_slice[client_idx]) for client_idx in range(client_num)}
    return dict_clients
