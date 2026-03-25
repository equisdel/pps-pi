# toma PF.json, lo procesa para calcular la redundancia por cada frente de pareto: por duplicado y por canonicidad, junto a la redundancia total
# mientras registra individuos únicos y computa el frente completo al final. la salida es: MERGED_PF, y tiene esta forma:

# ID | #MS |  CNT |    NED |      SM |     ICP |     IFN | PARTITION
# ----------------------------------------------------------------------------------------
# 0 |   8 |   19 |  0.750 |  0.7293 |  0.4130 |  1.8750 | [[0], [1, 2, 3, 5, 6, 8, 9, 10, 18, 19], [4, 11, 12, 13, 17, 21, 22, 23], [7], [14], [15], [16], [20]]

# entonces los outputs son: redundancy.txt y merged_pf.txt


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from redundancy.paths import PF_INPUT_PATH, REDUNDANCY_OUTPUT_PATH
from algorithm.initialization import progress_bar
import json
import ast
import csv

from collections import defaultdict

def unique_genotypes(pf):
    seen = set()
    uniq = []
    for ind in pf:
        t = tuple(ind)
        if t not in seen:
            seen.add(t)
            uniq.append(ind)
    return uniq

def canonical_partition(ind):
    blocks = defaultdict(list)
    for i, v in enumerate(ind):
        blocks[v].append(i)
    return frozenset(frozenset(b) for b in blocks.values())

def unique_canonicals(pf):
    seen = set()
    for ind in pf:
        seen.add(canonical_partition(ind))
    return seen


def compute_redundancy(idx,pf):

    pf_size = len(pf)

    # 1. Redundancia genotípica
    pf_geno = unique_genotypes(pf)
    geno_size = len(pf_geno)

    # 2. Redundancia por simetría
    pf_canon = unique_canonicals(pf)
    canon_size = len(pf_canon)

    # 3. Métricas
    red_dup = 1 - geno_size / pf_size
    red_sim = 1 - canon_size / geno_size if geno_size > 0 else 0
    red_total = 1 - canon_size / pf_size

    return [
        idx,
        pf_size,
        geno_size,
        canon_size,
        round(red_dup,4),
        round(red_sim,4),
        round(red_total,4)
    ]


def process_output(orig_path=PF_INPUT_PATH):

    with open(orig_path, "r") as f:
        pf_raw = json.load(f)

    # parse strings into real lists
    pf_parsed = {}
    redundancy_rows = []

    i = 0
    for key, pf_list in pf_raw.items():
        
        progress_bar(i,len(pf_raw))
        
        pf_parsed[key] = [ast.literal_eval(pf_str) for pf_str in pf_list]

        redundancy_rows.append(compute_redundancy(key,pf_list))

        i+=1

    redundancy_df = pd.DataFrame(redundancy_rows,columns=["index","pf_size","geno_size","canon_size","red_dup","red_sim","red_total"])
    redundancy_df.to_csv(REDUNDANCY_OUTPUT_PATH, index=False)

    return redundancy_df


if __name__=="__main__":
    
    pareto_fronts = process_output()

