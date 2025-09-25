import os
import sys
import collections
import json

# Last 3 digits of trial number
if len(sys.argv) > 1:
    trial_num = sys.argv[1]
'''
Enter trial number with 3 numbers in total, left padded by zeros when needed: i.e.: Trial 7 = 007, Trial 3 = 003, Trial 12 = 012
'''
# Filer address, where all data is located
if len(sys.argv) > 1:
    root = r'/mnt/research-projects/e/ejlobato/assist1data/cyber_plant/SideCam/Trial'+str(trial_num)
else:
    root = r'/mnt/research-projects/e/ejlobato/assist1data/cyber_plant/SideCam/Trial007'
    trial_num = '007'

diry = os.path.join(root)
count = 0
trial = {}
for (dirname, dirs, files) in os.walk(diry):
    for filename in files:
        # if jpg picture and date folder in root directory
        if filename.endswith('.jpg') and 'y' in dirname.split(os.sep)[9]:
            if dirname.split(os.sep)[9] in trial:
                trial[dirname.split(os.sep)[9]] += 1
            else:
                trial.update({dirname.split(os.sep)[9]: 1})
            count += 1
trial.update({'Total':count})
print('Total of images: {}'.format(trial['Total']))
od = collections.OrderedDict(sorted(trial.items()))
print(od)
with open(os.path.join(root, 'files_per_session_trial{}.json'.format(trial_num)), 'w') as fp:
    json.dump(od, fp, indent=4, sort_keys=True)