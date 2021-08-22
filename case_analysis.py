graph_set = {}

n_total = 0
n_new_two = 0
n_new_one = 0

temp = -5   ## compared with last N examples

e1_last = [-1]
e2_last = [-1]
n_same_last = 0

with open('case_graph_order.txt') as filein:
    for line in filein:
        fields = line.split('\t')
        if len(fields) < 5:
            continue
        e1_num, e1_id = fields[0], fields[1]
        e2_num, e2_id = fields[3], fields[4]

        if e1_num in e1_last[temp:] or e2_num in e1_last[temp:]: # or e1_num in e2_last[temp:] or e2_num in e2_last[temp:]:
            n_same_last += 1
        e1_last.append(e1_num)
        e2_last.append(e2_num)

        if e1_num not in graph_set and e2_num not in graph_set:
            n_new_two += 1
        elif e1_num not in graph_set or e2_num not in graph_set:
            n_new_one += 1
        else:
            n_total += 1
        
        graph_set[e1_num] = 1
        graph_set[e2_num] = 1

print(n_new_one, n_new_two, n_total)
print(n_same_last)


import random

n = 51

all_edges = []

for i in range(0, n):
    for j in range(i+1, n):
        all_edges.append((i, j))

all_edges_len = len(all_edges)

graph_set = {}

n_total = 0
n_new_two = 0
n_new_one = 0

e1_last = [-1, -1, -1, -1]
e2_last = [-1, -1, -1, -1]
n_same_last = 0

for _ in range(all_edges_len):
    t = random.choice(all_edges)
    all_edges.remove(t)

    e1_num, e2_num = t

    if e1_num in e1_last[temp:] or e2_num in e1_last[temp:]: # or e1_num in e2_last[temp:] or e2_num in e2_last[temp:]:
        n_same_last += 1
    e1_last.append(e1_num)
    e2_last.append(e2_num)
    
    if e1_num not in graph_set and e2_num not in graph_set:
        n_new_two += 1
    elif e1_num not in graph_set or e2_num not in graph_set:
        n_new_one += 1
    else:
        n_total += 1
    
    graph_set[e1_num] = 1
    graph_set[e2_num] = 1

print(n_new_one, n_new_two, n_total)
print(n_same_last)
