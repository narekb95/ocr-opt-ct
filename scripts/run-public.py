#!/bin/python3
import os
import subprocess
from time import time

outdir = 'output/'
def endswith(s, p):
    return s[-len(p):] == p

def get_all_instances(directory):
    folder = next(os.walk(directory, topdown=True))
    return folder[2]

def run_instance(directory, noar_dir, instance):
    input_file = f'{directory}{instance}.gr'
    sol_file = f'{outdir}{instance}.sol'
    start = time()
    subprocess.run(['./fre', input_file, sol_file])
    ellapsed_time = time() - start

    noar_file = f'{noar_dir}{instance}.gr'
    sol_verr_arguments = ['pace2024verifier', noar_file, sol_file, '--only-crossings']
    sol_verrifier = subprocess.run(sol_verr_arguments, stdout=subprocess.PIPE)
    sol_output = sol_verrifier.stdout.decode('utf8')[:-1]
    print(f'{instance:3}:\t{sol_output:2}\ttime: {ellapsed_time:5.2f}s')


dir = 'instances/public/'
noar_dir = 'instances/public/noar/'
instances = list(map(lambda instance: instance[:-len('.gr')],  get_all_instances(dir)))

def cmp (f1, f2):
    return int(f1 - f2)
instances.sort(key = lambda x : int(x))

all_data = []
for instance in instances:
    run_instance(dir, noar_dir, instance)