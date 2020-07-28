import numpy as np
import torch
import json

# save test
checkpoint = torch.load('/home/data/deecamp/baseline/Det3D/res/latest.pth')
checkpoint_json = dict()
checkpoint_json['meta'] = checkpoint['meta']
checkpoint_json['state_dict'] = dict()
for k, v in checkpoint['state_dict'].items():
    if isinstance(v, torch.Tensor):
        checkpoint_json['state_dict'][k] = v.numpy().tolist()
json_file = './res/latest.json'
with open(json_file, 'w') as fout:
    json.dump(checkpoint_json, fout)

# load test
json_file = './res/train_val_sim_all_laptop_epoch_50.json'
with open(json_file, 'r') as fin:
    checkpoint_json = json.load(fin)

    checkpoint = dict()
    checkpoint['meta'] = checkpoint_json['meta']
    checkpoint['state_dict'] = dict()
    for k, v in checkpoint_json['state_dict'].items():
        if isinstance(v, list):
            checkpoint['state_dict'][k] = torch.from_numpy(np.asarray(v, dtype=np.float32))

    # save to pth
    torch.save(checkpoint, './res/train_val_sim_all_laptop_epoch_50.pth')

print('finish!')
