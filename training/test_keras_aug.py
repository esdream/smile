import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 图片生成器ImageDataGenerator
# 用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止
# 参数
# featurewise_center: boolean, 使输入数据集去中心化（均值为0）
# samplewise_center: boolean, 使输入数据的每个样本均值为0
# feature_std_normalizetion: boolean, 将输入除以数据集的标准差以完成标准化，按feature执行
# zca_whitening: boolean, 对输入数据施加ZCA白化
# zca_epsilon: ZCA使用的eposilon(ε)，默认为1e-6
# rotation_range: 整数，数据提升时图片随机传动的角度（0-180度）
# width_shift_range: 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
# height_shift_range: 浮点数，图片高度的某个比例，数据提升时图片数值位移的幅度
# shear_range: 浮点数，剪切强度（逆时针方向的剪切变化角度）
# zoom_range: 浮点数或形如[lower, upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower, upper] = [1 - zoom_range, 1 + zoom_range]
# fill_mode: 当进行变化时超出边界的点将根据本参数给定的方法处理
# cval: 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
# horizontal_flip: boolean, 进行随机水平翻转
# vertical_flip: boolean, 进行随机竖直反转
# rescale: 重放缩因子，默认为None。如果为None或0则不进行放缩。否则会将该数值乘到数据上（在应用其他变换之前）
datagen = ImageDataGenerator(
        rescale=1.255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

i = 0
for batch in datagen.flow_from_directory(
    'keras_aug',
    target_size=(48, 48),
    batch_size=32,
    save_to_dir='keras_aug',
    save_prefix='aug'):
    i += 1
    if(i == 20):
        break

# for image_path in os.listdir('keras_aug'):
#     img = load_img(os.path.join('keras_aug', image_path))
#     img_source = img_to_array(img)
#     img_source = img_source.reshape((1, ) + img_source.shape)

# label = np.array(['sad'])
# # datagen.fit(img_source)

# i = 0
# for batch in datagen.flow(
#     img_source,
#     label,
#     batch_size=32,
#     save_to_dir='keras_aug',
#     save_prefix='aug',
#     save_format='png'):

#     i += 1
#     print(i)
#     if(i == 20):
#         break
