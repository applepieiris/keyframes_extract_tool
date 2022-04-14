from unicodedata import name
from TransNetV2.inference.transnetv2 import TransNetV2
import numpy as np
import os
import cv2
import random
import shutil

def shuffle_several_pic(scenes):
    idx_set = []
    for start, end in scenes :
        # 将首尾的frame先加进来
        idx_set.append(start)
        idx_set.append(end)
        lists = [i for i in range(start+1,end)]
        sample_num = (end-start-1) // 5 # 除去首尾两帧，中间随机取1/5帧作为关键帧 
        shuffle_set = random.sample(lists,sample_num)
        idx_set.extend(shuffle_set)
    idx_set.sort(reverse=True)
    return idx_set      
    
def save_keyframes(file_name,file,scenes):
        import cv2
        #predictions = (single_frame_predictions > 0.5).astype(np.uint8)
        cap = cv2.VideoCapture(file)
        frames_save_dir = os.path.join('./transnetv2_keyframes',os.path.splitext(file_name)[0])
        if not os.path.exists(frames_save_dir):
            os.makedirs(frames_save_dir)
        success , frame = cap.read()
        idx = 0
        keyframes_set = shuffle_several_pic(scenes)
        while success:
            if idx in keyframes_set:
                cv2.imwrite(os.path.join(frames_save_dir,(str(idx) + '.jpg')),frame)
                keyframes_set.remove(idx)
            idx += 1
            success , frame = cap.read()
        cap.release()

def predict_save_frames(video_dir):
    for file_name in os.listdir(video_dir):
        file_path = os.path.join(video_dir,file_name)
        model = TransNetV2()
        try:
            video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(file_path)
            scenes = model.predictions_to_scenes(single_frame_predictions)
            save_keyframes(file_name,file=file_path,scenes=scenes)
        except:
            with open("error_log.txt",'a') as log_file :
                log_file.write(file_path + " predict error!" + '\n')
            continue    

if __name__ == "__main__":
    frames_save_dir = './transnetv2_keyframes'
    if os.path.exists(frames_save_dir):
        shutil.rmtree(frames_save_dir)
    predict_save_frames(video_dir="../video_scrapy/videos")

