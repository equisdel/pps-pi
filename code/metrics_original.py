import itertools
import re
from itertools import combinations

import numpy as np


def ned(partitions):
    ned_sum = 0
    class_len = 0
    for cluster in partitions.values():
        size = len(cluster)
        class_len += size
        if 5 <= size <= 20:
            ned_sum += size

    ned_score = 1
    if class_len > 0 and ned_sum > 0:
        ned_score = ned_score - (ned_sum / class_len)

    return round(ned_score, 3)


def sm(ROOT, partitions, runtime_call_volume, result=None):
    #higher is better

    if result == None:
        result = partitions

    clusters = []

    for p in partition_class_bcs_assignment:
        clusters.append(partition_class_bcs_assignment[p]['classes'])

    nodes, call_volume = get_call_info(ROOT, runtime_call_volume)

    edges = []
    for c1, c2 in call_volume:
        edges.append((c1, c2, call_volume[(c1, c2)]))

    mq = 0
    for i, c0 in enumerate(clusters):
        mu = 0
        for edge in edges:
            if edge[0] in c0 and edge[1] in c0:
                mu += 1

        if mu == 0: continue

        eps = 0
        for j, c1 in enumerate(clusters):
            if i == j: continue

            for edge in edges:
                if edge[0] in c0 and edge[1] in c1:
                    eps += 1

        mq += 2. * mu / (2 * mu + eps)

    return round(mq, 3)


def ifn(ROOT, partition_class_bcs_assignment,runtime_call_volume, result=None):
    #lower is better

    if result == None:
        result = partition_class_bcs_assignment

    clusters = []

    for p in partition_class_bcs_assignment:
        clusters.append(partition_class_bcs_assignment[p]['classes'])

    nodes, call_volume = get_call_info(ROOT, runtime_call_volume)

    edges = []
    for c1, c2 in call_volume:
        edges.append((c1, c2))

    K = len(clusters)
    i = 0

    for c0, c1 in itertools.combinations(clusters, 2):
        for x, y in itertools.product(c0, c1):
            if (x, y) in edges or (y, x) in edges:
                i += 1

    return round(i * 1. / K, 3)


def icp(ROOT, class_bcs_partition_assignment, runtime_call_volume, result=None):
    """ The percentage of runtime call between two clusters. """
    #lower is better
    if result == None:
        result = class_bcs_partition_assignment

    n_total = 0
    n_inter = 0
    for call, volume in runtime_call_volume.items():
        src, target = call.split("--")
        if src.lower() == str(ROOT).lower() or target.lower() == str(ROOT).lower():
            continue

        if src == target:
            continue

        if src and target:

            src_assignment, target_assignment = result[src]['final'],  result[target]['final']
            n_total += np.log(volume) + 1
            if src_assignment != target_assignment:
                n_inter += np.log(volume) + 1

    try:
        r = n_inter * 1.0/n_total
    except ZeroDivisionError:
        r = float("Inf")

    return round(r,3)




def calculate(partitions, parsed_data):
    """
        clusters: dict of arrays { 0 : ['classA'], ...}
    """

    classes_to_ignore = set()

    """
    TODO IGNORING INHERITANCE
    for classes in partitions.values():
        for class_name in classes:
            classe_data = parsed_data[class_name]
            for extend in classe_data['extendedTypes']:
                classes_to_ignore.add(extend)
    """
    total_scoh = 0
    for cluster in partitions.values():
        total_scoh += scoh(cluster, parsed_data, classes_to_ignore)

    total_scop = 0
    for src, dst in combinations(partitions.keys(), 2):
        calc_scop = scop(src, dst, partitions, parsed_data, classes_to_ignore)
        total_scop += calc_scop

    N = len(partitions.keys())
    total_scoh = total_scoh / N
    total_scop = total_scop / (N * (N - 1) / 2)
    smq = total_scoh - total_scop

    return smq, total_scoh, total_scop


def scoh(cluster, parsed_classes, classes_to_ignore):
    cluster = set(cluster)

    edges = 0
    max_edges = 0
    for src, dst in combinations(cluster, 2):
        max_edges += 2  # bidirectional
        try:
            src_invocations = {method['targetClassName']
                               for method in parsed_classes[src]['methodInvocations']}
            dst_invocations = {method['targetClassName']
                               for method in parsed_classes[dst]['methodInvocations']}

            src_invocations = src_invocations | set(
                parsed_classes[src]['dependencies'])
            dst_invocations = dst_invocations | set(
                parsed_classes[dst]['dependencies'])

            # Check both directions
            if src in dst_invocations:
                edges += 1
            if dst in src_invocations:
                edges += 1

        except KeyError:
            print(f"[EXCEPTION KeyError] {src} or {dst} not found")

    if max_edges == 0:
        return 0
    print(
        f"SCOH: edges {edges} , len cluster: {len(cluster)}, scoh: {edges / (max_edges)}")
    return edges / (max_edges)


def scop(cluster_i, cluster_j, clusters, parsed_data, classes_to_ignore):
    classes_i = set(clusters[cluster_i])
    classes_j = set(clusters[cluster_j])

    # print(f"Classe_i {cluster_i} {classes_i}")
    # print(f"Classe_j {cluster_j} {classes_j}")

    total_edges = 0
    # Counts the number of edges between I and J
    for classe in classes_i:
        for method in parsed_data[classe]['methodInvocations']:
            if method['targetClassName'] in classes_j:
                total_edges += 1

    # Same as above, but inverse order
    for classe in classes_j:
        for method in parsed_data[classe]['methodInvocations']:
            if method['targetClassName'] in classes_i:
                total_edges += 1

    # print(
    #     f"SCOP: edges {total_edges},  scop: {total_edges / (2 * (len(classes_i) * len(classes_j)))}")
    return total_edges / (2 * (len(classes_i) * len(classes_j)))


def string_to_dict_arrays(string):
    clusters = string.split(":")[1:]
    processed_clusters = {}
    for index, c in enumerate(clusters):
        processed_clusters[index] = []
        arr = []
        c = c.strip()
        match = re.findall(r"'([a-zA-Z0-9._-]*)'", c)
        for m in match:
            processed_clusters[index].append(m)

    return processed_clusters


def static_cohesion(partitions, dependency_graph) -> float:

    N = max(partitions.values()) + 1
    # We want partitions to be 0, 1, ..., N-1 without any missing
    assert all(i in partitions.values() for i in range(N))
    assert len(set(partitions.values())) == N

    _internal_counts = np.zeros(N, dtype=np.float32)
    _external_counts = np.zeros(N, dtype=np.float32)

    for src_class, dst_class, data in dependency_graph.edges(data=True):

        if src_class not in partitions or dst_class not in partitions:
            continue

        weight = data['weight'] if 'weight' in data else 1.0

        src_partition = partitions[src_class]
        dst_partition = partitions[dst_class]

        if src_partition == dst_partition:
            _internal_counts[src_partition] += weight
        else:
            _external_counts[src_partition] += weight
            _external_counts[dst_partition] += weight

    cohesion = np.mean((2 * _internal_counts) / ((2 * _internal_counts) + _external_counts + 1e-7))

    return np.round(cohesion, 3)


def static_coupling(partitions, dependency_graph) -> float:

    intra_partition: int = 0
    extra_partition: int = 0

    for edges in dependency_graph.edges(data=True):
        src_class, dst_class, data = edges
        weight = data['weight'] if 'weight' in data else 1.0

        if (src_class not in partitions) or (dst_class not in partitions):
            if (src_class not in partitions) and (dst_class not in partitions):
                intra_partition += weight
            else:
                extra_partition += weight
            continue

        if partitions[src_class] == partitions[dst_class]:
            intra_partition += weight
        else:
            extra_partition += weight

    total   = extra_partition + intra_partition + 1e-7
    sipv    = extra_partition / float(total)

    return sipv
