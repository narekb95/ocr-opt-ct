#!/bin/python3
import os
import subprocess
def endswith(s, p):
    return s[-len(p):] == p

def get_all_instances(directory):
    folder = next(os.walk(directory, topdown=True))
    return folder[2]

# Verzeichnispfad, den Sie durchsuchen möchten
directory = 'instances/tiny'
instances = get_all_instances(directory)
all_data = []
for instance in instances:
    instance = instance[:-len('.gr')]
    input_file = f'{directory}/{instance}.gr'
    sol_file = f'{directory}/sol/{instance}.sol'
    arrg_file = f'{directory}/arrg/{instance}_arrg.gr'
    verrify_arguments = ['pace2024verifier', input_file, sol_file, '--only-crossings']
    verrifier_sol = subprocess.run(verrify_arguments, stdout=subprocess.PIPE)
    verrifier_out = verrifier_sol.stdout.decode('utf8')[:-1]
    mysol = subprocess.run(['./out', arrg_file],
                           stdout=subprocess.PIPE)
    my_out = mysol.stdout[:-1].decode('utf8')
    print(f'{verrifier_out:2} {my_out:2} {instance}')
    # print(instance, verrifier_sol.stdout, mysol.stdout)