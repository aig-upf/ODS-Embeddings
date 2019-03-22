import json

vals = [l.strip().split() for l in open('MovieLens.genres.csv')]
mapping = {t[0]: ' '.join(t[1:]).split('|') for t in vals}
values = {k: i for i, k in enumerate(sorted(set(t for v in mapping.values() for t in v if t[0] != '(')))}

def vectorize(targets):
    res = [0] * len(values)
    for v in targets:
        if v in values:
            res[values[v]] = 1
    return res

map_vectors = {k: vectorize(v) for k, v in mapping.items()}
with open('MovieLens.genres.json', 'w') as f:
    json.dump(map_vectors, f)

