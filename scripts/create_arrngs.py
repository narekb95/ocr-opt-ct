#!/bin/python3
import os
from sys import argv
from random import shuffle

def list_files(directory):
    return next(folder for folder in os.walk(directory, topdown=True))[2]

# Verzeichnispfad, den Sie durchsuchen möchten
directory = argv[1]
files = list_files(directory)
all_data = []
for file in files:
    file = file[:-len('.gr')]
    input_file = directory + file + '.gr'
    output_file = directory + '/arrg/' + file + '_arrg.gr'
    print(input_file, output_file)
    input_stream = open(input_file, 'r')
    input = input_stream.read().split('\n')
    output = open(output_file, 'w')
    for line in input:
        if len(line) > 0 and line[0] == 'p':
            line = line.split(' ')
            n1 = int(line[2])
            n2 = int(line[3])
            line.append(str(n1*n2))
            line = ' '.join(line)
            output.write(line+'\n')
            lin_arrange = list(range(n1+n2))
            # shuffle(lin_arrange)
            for i in lin_arrange:
                output.write(str(i+1)+'\n')
        else:
            if(len(line) > 0):
                output.write(line + '\n')