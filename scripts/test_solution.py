#!/bin/python3
import os
import subprocess
def endswith(s, p):
    return s[-len(p):] == p

def get_all_instances(directory):
    folder = next(os.walk(directory, topdown=True))
    return folder[2]

def run_instance(directory, instance):
    input_file = f'{directory}/{instance}.gr'
    sol_file = f'{directory}/sol/{instance}.sol'
    arrangement_file = f'{directory}/arrg/{instance}_arrg.gr'
    output_file = f'{directory}/arrg/{instance}.out'
    
    subprocess.run(['./fre', arrangement_file, output_file])


    sol_verr_arguments = ['pace2024verifier', input_file, sol_file, '--only-crossings']
    sol_verrifier = subprocess.run(sol_verr_arguments, stdout=subprocess.PIPE)
    sol_output = sol_verrifier.stdout.decode('utf8')[:-1]

    out_verr_arguments = ['pace2024verifier', input_file, output_file, '--only-crossings']
    out_verrifier = subprocess.run(out_verr_arguments, stdout=subprocess.PIPE)
    out_output = out_verrifier.stdout.decode('utf8')[:-1]
    os.remove(output_file)

    print(f'{sol_output:2} {out_output:2} {instance}')
    # print(instance, verrifier_sol.stdout, mysol.stdout)


directory = 'instances/tiny'
instances = get_all_instances(directory)
all_data = []
for instance in instances:
    run_instance(directory, instance[:-len('.gr')])