import os
import glob
root = '/apdcephfs/private_jiaweiyang/dataset/NCT/data/'
print(len(glob.glob(f'{root}/*/*')))

