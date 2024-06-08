#!/bin/python3
import os
from sys import argv
from random import shuffle, randint

def list_files(directory):
    return next(folder for folder in os.walk(directory, topdown=True))[2]

# Verzeichnispfad, den Sie durchsuchen möchten
directory = argv[1]
files = list_files(directory)
all_data = []
for file in files:
    file = file[:-len('.gr')]
    input_file = directory + file + '.gr'
    output_file = directory + 'arrg/' + file + '_arrg.gr'
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
            arr1 = list(range(1, n1 + 1))
            arr2 = list(range(n1 + 1, n1 + n2 + 1))
            shuffle(arr2)
            while(len(arr1) > 0 or len(arr2) > 0):
                r = randint(0, len(arr1) + len(arr2) - 1)
                if(r >= len(arr1)):
                    output.write(str(arr2.pop(0))+'\n')
                else:
                    output.write(str(arr1.pop(0))+'\n')
        else:
            if(len(line) > 0):
                output.write(line + '\n')