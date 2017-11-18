import os
import csv
import argparse
import numpy as np
from itertools import islice
from PIL import Image

# List of folders for training, validation and test.
folder_names = {'Training'   : 'FER2013Train',
                'PublicTest' : 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}

# 分类到不同表情的目录中
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear' ,' contempt', 'unknown', 'NF']

def str_to_image(image_blob):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)

def main(base_folder, fer_path, ferplus_path):
    '''
    Generate PNG image files from the combined fer2013.csv and fer2013new.csv file. The generated files
    are stored in their corresponding folder for the trainer to use.
    
    Args:
        base_folder(str): The base folder that contains  'FER2013Train', 'FER2013Valid' and 'FER2013Test'
                          subfolder.
        fer_path(str): The full path of fer2013.csv file.
        ferplus_path(str): The full path of fer2013new.csv file.
    '''
    
    print("Start generating ferplus images.")
    
    for key, value in folder_names.items():
        folder_path = os.path.join(base_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for emotion in emotions:
            emotion_path = os.path.join(folder_path, emotion)
            if not os.path.exists(emotion_path):
                os.makedirs(emotion_path)

    ferplus_entries = []
    with open(ferplus_path,'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        # islice(iterable, start, stop[, step])返回一个迭代器，在传入的可迭代对象中，从索引为start的元素开始，到stop结束（stop为None时到结尾），按step选取元素，并传入到返回的迭代器里。以下是把ferplus_rows的首行去掉，迭代至末尾为止
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)
 
    index = 0
    with open(fer_path,'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                # 将字符串转化为image对象
                image = str_to_image(row[1])
                
                tags = [int(tag) for tag in ferplus_row[2:]]
                emotion = tags.index(max(tags))

                # 图像存储的路径，其中folder_name[row[2]]是Train, Valid和Test, emotions[emotion]是每个表情的目录
                image_path = os.path.join(
                    base_folder, folder_names[row[2]], emotions[emotion], file_name)
                image.save(image_path, compress_level=0)                
            index += 1 
            
    print("Done...")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type = str, 
                        help = "Base folder containing the training, validation and testing folder.", 
                        required = True)
    parser.add_argument("-fer", 
                        "--fer_path", 
                        type = str,
                        help = "Path to the original fer2013.csv file.",
                        required = True)
                        
    parser.add_argument("-ferplus", 
                        "--ferplus_path", 
                        type = str,
                        help = "Path to the new fer2013new.csv file.",
                        required = True)                        

    args = parser.parse_args()
    main(args.base_folder, args.fer_path, args.ferplus_path)
