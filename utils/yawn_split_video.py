import os
import cv2
import numpy as np

NUM_ELEMENT = 3
NUM_VIDEO_CLASS = 4
TAGS = ['Normal', 'Talking', 'Yawning']
DATA_SET_LIST = ['train_lst', 'test_lst']
for data_set in DATA_SET_LIST:
    split_file_list = ['../dataset/' + data_set + '/male_yawn_split.lst',
                       '../dataset/' + data_set + '/female_yawn_split.lst',
                       '../dataset/' + data_set + '/dash_female_yawn_split_yawning.lst',
                       '../dataset/' + data_set + '/dash_male_yawn_split_yawning.lst']

    dataset_path_list = ['../YawDD_dataset/Mirror/Male_mirror',
                         '../YawDD_dataset/Mirror/Female_mirror',
                         '../YawDD_dataset/Dash/Female',
                         '../YawDD_dataset/Dash/Male']

    save_path_list = ['../dataset/' + data_set + '/mirror_male_split_output',
                      '../dataset/' + data_set + '/mirror_female_split_output',
                      '../dataset/' + data_set + '/dash_female_split_output',
                      '../dataset/' + data_set + '/dash_male_split_output']
    for video_index in range(NUM_VIDEO_CLASS):

        split_file = split_file_list[video_index]
        dataset_path = dataset_path_list[video_index]
        save_path = save_path_list[video_index]

        for i in range(len(TAGS)):
            path = os.path.join(save_path, TAGS[i])
            if os.path.exists(path) is False:
                os.makedirs(path)

        print('split file: ', split_file)
        fd = open(split_file)

        for line in fd.readlines():
            line = line.strip('\n')
            line = line.split(' ')
            video_file = os.path.join(dataset_path, line[0])
            num_clips = int((len(line) - 1) / NUM_ELEMENT)
            clips_info = np.array(map(int, line[1:]))
            clips_info = clips_info.reshape(num_clips, NUM_ELEMENT)

            cap = cv2.VideoCapture(video_file)
            fcnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fsize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(video_file)
            for i in range(num_clips):
                start_slice = clips_info[i, 0]
                fstart = int(start_slice * fps)
                # if -1, continue go to end
                if clips_info[i, 1] > 0:
                    end_slice = clips_info[i, 1]
                    fend = min(int(end_slice * fps), fcnt)
                else:
                    fend = int(fcnt) - 20
                if fstart >= fend:
                    print('=======start > end, continue======')
                    continue
                print('%d --> %d, tag: %s' % (fstart, fend, TAGS[clips_info[i, 2]]))
                assert(fstart > 0 or fend <= fcnt)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fstart)

                [_, file_name] = os.path.split(video_file)
                [video_name, _] = os.path.splitext(file_name)
                clip_file = '%s/%s/%s-clip-%d.avi' % (save_path, TAGS[clips_info[i, 2]], video_name, i)
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                writer = cv2.VideoWriter(clip_file, fourcc, fps=fps, frameSize=fsize)
                for n in range(fend-fstart):
                    success, img = cap.read()
                    writer.write(img)
                    #cv2.imshow('video', img)
                    #cv2.waitKey(40)
                writer.release()
                #cv2.waitKey(-1)
