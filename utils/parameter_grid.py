"""
This script is used for generating config files, a bit messy code.
"""

from sklearn.model_selection import ParameterGrid
import json
import os

# configuration folder, generated config files will be stored there
DIR = 'configs'

parameter_dict = {
    "version": [1],  # model version
    "num_epochs": [192],
    "num_iter_per_epoch": [18],
    "learning_rate": [1e-4],  # tested a lot, no need to change
    "batch_size": [10],  # can adjust this, but need more memory
    "state_size": [[128, 128, 1]],  # iamge size
    "max_to_keep": [1],  # kept model weights, don't change
    "is_training": [1],  # for batch norm, don't change
    "activation": ['prelu'],
    "pooling": ['max'],
    "target": ['Lumen', 'Media'],  # task
    "run": range(1, 6)  # how many models we want for each
}

grid = sorted(list(ParameterGrid(parameter_dict)), key=lambda x: x['target'])

try:  # create the DIR if not existeds
    os.stat(DIR)
except:
    os.mkdir(DIR)

for i in grid:
    i['exp_name'] = 'V{}-Journal-ImageSize{}-Epoch{}-Iter{}-LR{}-BS{}-{}-{}-{}-{}'.format(
        i['version'], i['state_size'][0], i['num_epochs'],
        i['num_iter_per_epoch'], i['learning_rate'], i['batch_size'],
        i['activation'], i['pooling'], i['target'], i['run'])

    # create config files
    with open('./configs/{}.json'.format(i['exp_name']), 'w') as f:
        print('./configs/{}.json'.format(i['exp_name']))
        json.dump(i, f)

    # create scripts for training
    # note "python3.5" is used here, may need to change
    with open('./mains/run.sh', 'a') as f:
        f.write('python main{}.py -c ../configs/{}.json\n'.format(
            i['version'], i['exp_name']))
