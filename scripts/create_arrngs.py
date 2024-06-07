import os

def list_files(directory):
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

# Verzeichnispfad, den Sie durchsuchen möchten
directory = 'tiny/arrg'
files = list_files(directory)
all_data = []
for file in files:
    input_stream = open(file, 'r')
    input = input_stream.read().split('\n')
    output_file = file[:len(file)-len('.gr')]+'_arrg.gr'
    print(output_file)
    output = open(output_file, 'w')
    for line in input:
        if len(line) > 0 and line[0] == 'p':
            line = line.split(' ')
            n1 = int(line[2])
            n2 = int(line[3])
            line.append(str(n1+n2))
            line = ' '.join(line)
            output.write(line+'\n')
            for i in range(n1+n2):
                output.write(str(i+1)+'\n')
            continue
        output.write(line + '\n')

    input_stream.close()
    os.remove(file)
