# toma PF.json, lo procesa para calcular la redundancia por cada frente de pareto: por duplicado y por canonicidad, junto a la redundancia total
# mientras registra individuos únicos y computa el frente completo al final. la salida es: MERGED_PF, y tiene esta forma:

# ID | #MS |  CNT |    NED |      SM |     ICP |     IFN | PARTITION
# ----------------------------------------------------------------------------------------
# 0 |   8 |   19 |  0.750 |  0.7293 |  0.4130 |  1.8750 | [[0], [1, 2, 3, 5, 6, 8, 9, 10, 18, 19], [4, 11, 12, 13, 17, 21, 22, 23], [7], [14], [15], [16], [20]]

# entonces los outputs son: redundancy.txt y merged_pf.txt


import os
import json
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
from algorithm.initialization import progress_bar


def normalize_individual(ind):
    if ind is None:
        return ()
    if isinstance(ind, str):
        raise ValueError("Individual must be a parsed list, not a raw string.")

    if isinstance(ind, list) and ind and all(isinstance(x, int) for x in ind):
        return tuple(ind)

    normalized_blocks = []
    for block in ind:
        if block is None:
            continue
        if isinstance(block, (list, tuple, set)):
            normalized_blocks.append(tuple(sorted(block)))
        elif isinstance(block, int):
            normalized_blocks.append((block,))
        else:
            normalized_blocks.append(tuple(sorted(list(block))))

    return tuple(sorted(normalized_blocks))


def unique_genotypes(pf):
    seen = set()
    uniq = []
    for ind in pf:
        normalized = normalize_individual(ind)
        if normalized not in seen:
            seen.add(normalized)
            uniq.append(ind)
    return uniq


def canonical_partition(ind):
    if isinstance(ind, list) and ind and all(isinstance(x, int) for x in ind):
        blocks = defaultdict(list)
        for i, v in enumerate(ind):
            blocks[v].append(i)
        return frozenset(frozenset(b) for b in blocks.values())

    blocks = []
    for part in ind:
        if part is None:
            continue
        blocks.append(frozenset(part))
    return frozenset(blocks)


def unique_canonicals(pf):
    seen = set()
    for ind in pf:
        seen.add(canonical_partition(ind))
    return seen


def compute_redundancy(idx, pf):
    pf_size = len(pf)

    pf_geno = unique_genotypes(pf)
    geno_size = len(pf_geno)

    pf_canon = unique_canonicals(pf)
    canon_size = len(pf_canon)

    red_dup = 1 - geno_size / pf_size if pf_size > 0 else 0
    red_sim = 1 - canon_size / geno_size if geno_size > 0 else 0
    red_total = 1 - canon_size / pf_size if pf_size > 0 else 0

    return [
        idx,
        pf_size,
        geno_size,
        canon_size,
        round(red_dup, 4),
        round(red_sim, 4),
        round(red_total, 4),
    ]


def process_output(orig_path=None, output_path=None):
    with open(orig_path, "r", encoding="utf-8") as f:
        pf_raw = json.load(f)

    pf_parsed = {}
    redundancy_rows = []

    i = 0
    for key, pf_list in pf_raw.items():
        progress_bar(i, len(pf_raw))

        pf_parsed[key] = [ast.literal_eval(pf_str) for pf_str in pf_list]
        redundancy_rows.append(compute_redundancy(key, pf_parsed[key]))
        i += 1

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    redundancy_df = pd.DataFrame(
        redundancy_rows,
        columns=["index", "pf_size", "geno_size", "canon_size", "red_dup", "red_sim", "red_total"],
    )
    redundancy_df.to_csv(output_path, index=False)

    return redundancy_df


if __name__=="__main__":
    
    pareto_fronts = process_output()

