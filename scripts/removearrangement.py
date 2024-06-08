#!/bin/python3
import os
from sys import argv
from random import shuffle, randint

def list_files(directory):
    return next(folder for folder in os.walk(directory, topdown=True))[2]

# Verzeichnispfad, den Sie durchsuchen möchten
directory = argv[1]
out_dir = argv[2]
files = list_files(directory)
all_data = []
for file in files:
    file = file[:-len('.gr')]
    print(file)
    input_file = directory + file + '.gr'
    output_file = out_dir +  file + '.gr'
    input_stream = open(input_file, 'r')
    input = input_stream.read().split('\n')
    line = input[0].split(' ')
    n = int(line[2]) + int(line[3])
    print(n)
    out_list = [input[0]]
    out_list.extend(input[n+1:])
    print(len(out_list))
    text = '\n'.join(out_list)
    open(output_file, 'w').write(text)