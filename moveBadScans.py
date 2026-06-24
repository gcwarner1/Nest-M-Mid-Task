import os, re, sys, shutil

'''
This script finds scans that for whatever reason had to be stopped prematurely and moves them to a seperate location.
We then take the redone version of that scan and rename it in accoradance with our formating requirements.
This allows us to carry on with analyses using only the completed version of the scan.
'''

data = '/Users/braveDP/Desktop/NEST-M/NESTM_bids/'
out = '/Users/braveDP/Desktop/NEST-M/IncompleteNESTMscans/'
if os.path.exists(out) == False:
      sys.exit('No outdir')

for root, dirs, files in os.walk(data):
    for f in files:
            if 'redo' in f:
                  good = os.path.join(root, f)
                  bad = good.replace('redo','')
                  name = os.path.split(bad)[-1]
                  print('\nFirst')
                  print(bad+'----->'+out+name)
                  shutil.move(bad, out+name)
                  print('\nSecond')
                  print(good+'----->'+os.path.join(root,name))
                  shutil.move(good, os.path.join(root,name))
print('Finished')
