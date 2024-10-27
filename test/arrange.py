import os
from tqdm import tqdm


root = 'test/leadsheet_to_multi-track/demo_3'
for folder in tqdm(os.listdir(root)):
    subroot = os.path.join(root, folder)
    os.rename(subroot, os.path.join(root, folder.split('-')[0]))
    
