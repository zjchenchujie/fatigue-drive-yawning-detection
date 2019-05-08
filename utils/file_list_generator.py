"""
This script split dataset into four sets: test.txt, trainval.txt, train.txt, val.txt
"""

import os

import random


class ListGenerator:
    """Generate a list of specific files in directory."""

    def __init__(self):
        """Initialization"""
        # The list to be generated.
        self.file_list = []

    def generate_list(self, src_dir, save_dir, format_list=['jpg', 'png'], train_val_ratio=0.9, train_ratio=0.9):
        """Generate the file list of format_list in target_dir

        Args:
            target_dir: the directory in which files will be listed.
            format_list: a list of file extention names.

        Returns:
            a list of file urls.

        """
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.format_list = format_list
        self.train_val_ratio = train_val_ratio
        self.train_ratio = train_ratio

        # Walk through directories and list all files.
        pos_num = 0
        for file_path, _, current_files in os.walk(self.src_dir, followlinks=False):
            current_image_files = []
            for filename in current_files:
                # First make sure the file is exactly of the format we need.
                # Then process the file.
                if filename.split('.')[-1] in self.format_list:
                    current_image_files.append(filename)

            sample_num = len(current_image_files)
            file_list_index = range(sample_num)
            tv = int(sample_num * self.train_val_ratio)
            tr = int(tv * self.train_ratio)
            print("train_val_num: {}\ntrain_num: {}".format(tv, tr))
            train_val = random.sample(file_list_index, tv)
            train = random.sample(train_val, tr)

            ftrain_val = open(self.save_dir + '/trainval.txt', 'w')
            ftest = open(self.save_dir + '/test.txt', 'w')
            ftrain = open(self.save_dir + '/train.txt', 'w')
            fval = open(self.save_dir + '/val.txt', 'w')
            for i in file_list_index:
                label = current_image_files[i].split('.')[0]
                label = label.split('_')[-1]
                if label == '1':
                    pos_num += 1
                name = os.path.join(self.src_dir, current_image_files[i]) + ' ' + str(label) + '\n'
                if i in train_val:
                    ftrain_val.write(name)
                    if i in train:
                        ftrain.write(name)
                    else:
                        fval.write(name)
                else:
                    ftest.write(name)
            print('positive num : {}'.format(pos_num))
            ftrain_val.close()
            ftrain.close()
            fval.close()
            ftest.close()

        return self.file_list

def main():
    """MAIN"""
    lg = ListGenerator()
    lg.generate_list(src_dir='../extracted_face/face_image',
                     save_dir='../extracted_face')
    print("Done !!")
    # lg.save_list()


if __name__ == '__main__':
    main()
