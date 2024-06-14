import os
root_path = '/mnt/d/SimpsonLab/PDAC response/PDAC response/ImagingData/Pre-treatment/'
for folder in os.listdir(root_path):
    # os.mkdir(f'{root_path}{folder}/mha')
    for file in os.listdir(f'{root_path}/{folder}'):
        # print(f'{root_path}{folder}/{file}')
        if file[-3:]=='mha' and not os.path.isdir(f'{root_path}{folder}/{file}'):
            os.rename(f'{root_path}{folder}/{file}',f'{root_path}{folder}/mha/{file}')