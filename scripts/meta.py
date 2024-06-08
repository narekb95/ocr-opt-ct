#!/bin/python3
import os

def list_files(directory):
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

# Verzeichnispfad, den Sie durchsuchen möchten
directory = 'instances/public'
files = list_files(directory)
all_data = []
for file in files:
    input = open(file, 'r').read().split('\n')
    first_line = next(line for line in input if line[0] == 'p').split()
    all_data.append({
        'n1': int(first_line[2]),
        'n2': int(first_line[3]),
        'm': int(first_line[4]),
        'ctw': int(first_line[5]),
    })

n1 = list(map(lambda x: x['n1'], all_data))
n2 = list(map(lambda x: x['n2'], all_data))
m = list(map(lambda x: x['m'], all_data))
ctw = list(map(lambda x: x['ctw'], all_data))
print(max(n1), max(n2), max(m), max(ctw))

ctw_finer = [(x*10, len([c for c in ctw if c/10 < x])) for x in range(10)]
print(ctw_finer)