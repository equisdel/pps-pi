from collections import defaultdict

def ned(partitions,n_classes=None):
    k = len(partitions)
    if k == 0:
        return 1  # Max penalty if no microservices
    # entre el 5 y el 40 por ciento de las clases (salvo instancias grandes)
    lower_limit = min(int((n_classes/100)*5)+1, 5) if n_classes else 5   # 1
    upper_limit = min(int((n_classes/100)*40), 20) if n_classes else 20  # 9

    non_extreme_count = sum(1 for cluster in partitions.values() if lower_limit < len(cluster) < upper_limit)
    ned_value = 1 - (non_extreme_count / k)
    return round(ned_value, 3)


def ifn(partitions, graph):
    class_to_microservice = {}
    for microservice_id, microservice in partitions.items():
        for cls in microservice:
            class_to_microservice[cls] = microservice_id

    interface_classes_per_ms = {ms_id: set() for ms_id in partitions}

    for caller, callee in graph.edges:
        ms_caller = class_to_microservice.get(caller)
        ms_callee = class_to_microservice.get(callee)
        if ms_caller != ms_callee and ms_callee in interface_classes_per_ms:
            interface_classes_per_ms[ms_callee].add(callee)

    total_interfaces = sum(len(classes) for classes in interface_classes_per_ms.values())
    k = len(partitions)
    return total_interfaces / k if k > 0 else 0


def sm(partitions, graph):
    K = len(partitions)
    if K < 2:
        return 0  # Avoid division by zero and meaningless SM

    # Create a reverse map: class -> microservice ID
    class_to_ms = {}
    for ms_id, classes in partitions.items():
        for cls in classes:
            class_to_ms[cls] = ms_id

    # Initialize intra and inter-service call counts
    mu = defaultdict(int)  # intra-service call count per microservice
    sigma = defaultdict(lambda: defaultdict(int))  # inter-service call count between microservices

    for c_i, c_j in graph.edges:
        ms_i = class_to_ms.get(c_i)
        ms_j = class_to_ms.get(c_j)

        if ms_i is None or ms_j is None:
            continue  # skip unassigned classes

        if ms_i == ms_j:
            mu[ms_i] += 1  # internal call
        else:
            sigma[ms_i][ms_j] += 1  # external call from ms_i to ms_j

    # First term: cohesion (intra-service)
    cohesion_sum = 0
    for ms_id, classes in partitions.items():
        m_i = len(classes)
        if m_i > 0:
            cohesion_sum += mu[ms_id] / (m_i * m_i)

    cohesion_term = cohesion_sum / K

    # Second term: coupling (inter-service)
    coupling_sum = 0
    ms_ids = list(partitions.keys())
    for i in range(K):
        for j in range(i + 1, K):
            ms_i = ms_ids[i]
            ms_j = ms_ids[j]
            m_i = len(partitions[ms_i])
            m_j = len(partitions[ms_j])
            if m_i == 0 or m_j == 0:
                continue
            s_ij = sigma[ms_i][ms_j] + sigma[ms_j][ms_i]
            coupling_sum += s_ij / (2 * m_i * m_j)

    denominator = (K * (K - 1)) / 2
    coupling_term = coupling_sum / denominator if denominator != 0 else 0

    return round(cohesion_term - coupling_term, 4)

def icp(partitions, graph):
    total_dependencies = graph.number_of_edges()
    if total_dependencies == 0:
        return 0  # Avoid division by zero

    # Map each class to its corresponding microservice
    class_to_microservice = {}
    for microservice_id, microservice in partitions.items():
        for cls in microservice:
            class_to_microservice[cls] = microservice_id

    inter_service_calls = sum(
        1 for c_i, c_j in graph.edges if class_to_microservice[c_i] != class_to_microservice[c_j]
    )

    icp_value = inter_service_calls / total_dependencies
    return icp_value

def hv():
    return hv


if __name__=="__main__":

    ind = [9, 2, 0, 18, 9, 13, 12, 16, 11, 4, 23, 6, 8, 14, 0, 7, 19, 14, 0, 3, 2, 22, 15, 13]
    print()
    print(ned(ind))
