import os
import shutil
import json

main_dir = '/home/jiaxuzhu/data/CUB_200_2011/CUB_200_2011'
image_dir = os.path.join(main_dir, 'images')
dst_dir = '/home/jiaxuzhu/data/cub_noisy'

classes = {}
bings = []

with open(os.path.join(main_dir, 'classes.txt'), 'r') as classFile:
    for line in classFile.readlines():
        tmp = line.split(' ')
        label = int(tmp[0])
        name = tmp[1]
        classes[label] = name.strip('\n')
        bings.append(classes[label].split('.')[-1].replace('_', ' '))

with open(os.path.join(dst_dir, 'bing_keywords.json'), 'w') as bingFile:
    json.dump(bings, bingFile)

with open(os.path.join(dst_dir, 'name_lst.json'), 'w') as nameLst:
    json.dump(classes, nameLst)

#
# is_train = []
# with open(os.path.join(main_dir, 'train_test_split.txt'), 'r') as splitFile:
#     for line in splitFile.readlines():
#         is_train.append(int(line.split(' ')[-1].strip('\n')))
#
# with open(os.path.join(main_dir, 'images.txt'), 'r') as imageFile:
#     images = imageFile.readlines()
#
# with open(os.path.join(main_dir, 'image_class_labels.txt'), 'r') as imageClass:
#     image_class = imageClass.readlines()
#
#
# if not os.path.exists(dst_dir):
#     os.mkdir(dst_dir)
#
# train_dir = os.path.join(dst_dir, 'clean_train')
# if not os.path.exists(train_dir):
#     os.mkdir(train_dir)
#
# val_dir = os.path.join(dst_dir, 'clean_val')
# if not os.path.exists(val_dir):
#     os.mkdir(val_dir)
#
# print len(images)
# for i in range(len(images)):
#     tmp = images[i].split(' ')
#     img_idx = int(tmp[0])
#     img_path = tmp[1].strip('\n')
#
#     tmp = image_class[i].split(' ')
#     img_class = classes[int(tmp[1])]
#     src = os.path.join(image_dir, img_path)
#     if is_train[i] == 1:
#         dst = train_dir
#     else:
#         dst = val_dir
#     if not os.path.exists(os.path.join(dst, img_class)):
#         os.mkdir(os.path.join(dst, img_class))
#
#     dst = os.path.join(dst, img_path)
#     shutil.copy2(src, dst)
#     # break

