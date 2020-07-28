filenames = ['/home/data/deecamp/DeepCamp_Lidar/sim_scenes_20200722/label/sim_val.txt',
             '/home/data/deecamp/DeepCamp_Lidar/sim_scenes_20200723/label/sim_train.txt',
             '/home/data/deecamp/DeepCamp_Lidar/sim_scenes_aug_20200724/labels_filer/sim_train_aug.txt',
             '/home/data/deecamp/DeepCamp_Lidar/sim_scenes_aug_20200725/labels_filer/sim_train_aug_5000_6000.txt']
with open('/home/data/deecamp/DeepCamp_Lidar/sim_scenes_all_part1_20200725/labels_filer/sim_all_filter.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)