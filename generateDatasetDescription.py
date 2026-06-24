#!/Users/braveDP/.conda/envs/bin/python

'''
This script generates dataset_description.json to comply with BIDS formatting. This script takes two arguments given in the terminal. 
The first is the full path to the bids directory in question (e.g. /Users/Desktop/NEST-M/NESTM_bids)
The second is the name of the study which is used to populate some of the json fields (e.g. NestM)

Usage:
    python generateDatasetDescription.py /path/to/bids/dir StudyName
'''


import mne_bids, sys, os

dataset=sys.argv[1] #e.g. /Users/braveDP/Desktop/NESTM_bids
studyName=sys.argv[2] #e.g. NestM

if os.path.isdir(dataset) == False:
    sys.exit('First input (path to data) is not a valid directory')

mne_bids.make_dataset_description(path=dataset, name=studyName)
print('Created dataset_description.json for '+dataset+' named '+studyName)