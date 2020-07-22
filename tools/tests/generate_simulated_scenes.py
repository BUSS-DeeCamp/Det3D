import json
import pickle
from pathlib import Path
from multiprocessing import Pool

from loguru import logger as logging
from tools.tests.object_utils import Box, ObjectWithBox
from tools.tests.random_scene_generation import SceneGenerator


class Generator(object):
    def __init__(self, data_folder, output_folder, objects_data):
        # create scene generator
        self.sg = SceneGenerator(data_folder, output_folder, objects_data)

    def run(self, data_dict):
        # generate a scene
        self.sg.generate_scene(data_dict)

        # save scene cloud to .bin file
        self.sg.save_scene_cloud_to_file()

        # save scene labels to .txt file
        self.sg.save_scene_labels_to_file()

        # get labels for the scene
        scene_labels = self.sg.get_scene_labels()

        # clean
        self.sg.reset()

        return scene_labels


if __name__ == '__main__':

    data_folder = '/home/data/deecamp/DeepCamp_Lidar'
    objects_folder = '/home/data/deecamp/DeepCamp_Lidar/objects'
    label_file = Path(data_folder).joinpath('labels_filer/val_filter.txt')

    output_scenes_folder = Path(data_folder).joinpath('sim_scenes')
    output_scenes_folder.mkdir(exist_ok=True)

    output_label_folder = Path(data_folder).joinpath('sim_scenes/label')
    output_label_folder.mkdir(exist_ok=True)

    # load objects
    object_files = Path(objects_folder).glob('*.bin')
    objects_data = list()
    for file in object_files:
        with open(file.as_posix(), 'rb') as f:
            objects_data.append(pickle.load(f))

    # load labels
    labels = []
    with open(label_file) as rf:
        for line in rf:
            data_dict = json.loads(line.strip())
            labels.append(data_dict)

    logging.info('Total labeled scenes: {}'.format(len(labels)))

    # generate and save results
    g = Generator(data_folder, output_scenes_folder, objects_data)
    p = Pool()
    generated_labels = p.map(g.run, labels)

    # save labels to file
    output_label_file = output_label_folder.joinpath('sim_scene.txt')
    with open(output_label_file.as_posix(), 'w') as fout:
        for label in generated_labels:
            fout.write(json.dumps(label))
            fout.write('\n')

    logging.info('Saved scene labels to file: {}'.format(output_label_folder.as_posix()))
