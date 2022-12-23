"""
@author:CS
@time:2022/8/16:10:13
"""
import numpy as np
import pandas as pd
import mak_models as mak
import time
import os

import warnings
warnings.filterwarnings('ignore')

logo = pd.read_pickle('mak')
for i in range(23):
    p = ''
    for j in range(68):
        p = p + logo.iloc[i, j] + ' '
    print(p)

logo.to_pickle('mak')


def check_path(work_path):
    try:
        os.chdir(work_path)
        print('Your working path was: ', work_path)
    except:
        print('Error: Unable to locate the specified folder, please check and input again !!!')
        work_path = input('Input your working path: ')
        check_path(work_path)


next_trait = True
while next_trait:
    time_start = time.time()
    work_path = input('Input your working path: ')
    check_path(work_path)
    species = input('Input species: ')
    trait = input('Input trait: ')
    multi_acc, single_acc= mak.calculate_multi_traits(species, trait)
    time_end = time.time()
    print('Trait: %s, MAK accuracy: %.3f, Single accuracy: %.3f , Time using: %.1f s' % (trait, multi_acc, single_acc, time_end - time_start))
    exit_y_n = input('quit (y/n):')
    if exit_y_n == 'y':
        next_trait = False








