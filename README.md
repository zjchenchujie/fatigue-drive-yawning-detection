# fatigue-drive-yawning-detection
This repository is to create a fatigue drive detection algorithm via driver yawning recognition

## 1. Demo
```python demo.py```

## 2. Train
### * Dataset Downloading
We use YawDD as our train dataset. This dataset contains two video datasets of drivers with various facial characteristics.The videos are taken in real and varying illumination conditions. You can download from [here](http://www.site.uottawa.ca/~shervin/yawning/).  

- In the first dataset, a camera is installed under the front mirror of the car.   
- In the second dataset, the camera is installed on the driver’s dash.   

### * Data Proprocessing
There are a list of driver videos contained in dataset. We randomly select some video that contains driver yawning and label the video fragment with 0-Normal, 1-Talking, 2-Yawning.  
For example:  
`
23-FemaleNoGlasses-Talking&Yawning.avi 0 7 1 8 9 2 10 -1 0
`  
means that: in video `23-FemaleNoGlasses-Talking&Yawning.avi`, 0s - 7s driver Talking (label:1), 8s - 9s driver Yawning (label:2), 10s - end driver normal (label: 0). Note that -1 means the end of video.  
- Note: We treated 'Talking' and 'Normal' as same, which means 'non-yawning' in this project.  

Divide the video into train_lst and test_lst sets. Make sure you are in directory `utils`:  
```
python yawn_split_video.py  
python extract_face_from_video.py
python file_list_generator.py
```
The scripts above will create `extracted_face` which contains driver face image directory `face_image` and four dataset lists: `est.txt, trainval.txt, train.txt, val.txt`
### * Start Training
Change your working directory to `model` and run `bash train.sh`
- Note: If necessary, you should change your `caffe` path and data source path in `train.sh` and `train.prototxt` as well as `test.prototxt`.
