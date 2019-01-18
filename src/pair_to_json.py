import sys
import json
import fileinput


sep = sys.argv[1]
out = sys.argv[2]
label_mapping = {}
for line in sys.stdin:
    n, l = line.strip().split(sep)
    label_mapping[int(n)] = int(l)

json.dump(label_mapping, open(out, 'w'))
