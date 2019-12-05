"""
针对目标检测文件解析数据供模型使用
"""
import numpy as np
import sys
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
from PIL import Image
import pickle
import sklearn.utils
from copy import deepcopy
from utils import BoxFilter

class DataGenerator:
    def __init__(self,
                 load_images_into_memory=False,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        '''
        参数:
            load_images_into_memory (bool, optional): 如果为True，则整个数据集将被加载到内存中。
            filenames (string or list, optional): “ None”或Python列表/元组或代表文件路径的字符串。
            filenames_type (string, optional): 如果为`filenames`传递了一个字符串，则表明文件`filenames`是什么类型。
            images_dir (string, optional): 如果为“文件名”传递了文本文件，则图像的完整路径将由“ images_dir”和
            文本文件中的名称组成。
            labels (string or list, optional): 标签。
            image_ids (string or list, optional): 图片ID。
            labels_output_format (list, optional): 标签输出格式。
            verbose (bool, optional): 是否打印进度。
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')}

        self.dataset_size = 0 # 只要我们还没有加载任何东西，数据集的大小就为零。
        self.load_images_into_memory = load_images_into_memory
        self.images = None # 该列表不保留为“ None”的唯一方法是“ load_images_into_memory == True”。

        # self.filenames是一个列表，其中包含图像样本的所有文件名（完整路径）。
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else: it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # 如果有真值, self.labels是一个列表，其中包含每个图像的真值边界框列表（或NumPy数组）。
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes='all',
                  verbose=True):
        '''
        这是一个针对Pascal VOC数据集格式的解析器。
        参数:
            images_dirs (list): 一个字符串列表，其中每个字符串是目录的路径，该目录包含将成为数据集一部分的图像。
            image_set_filenames (list): 一个字符串列表， 其中每个字符串是文本文件的路径，其中设置了要加载的图像。
            annotations_dirs (list, optional): 一个字符串列表， 其中每个字符串是包含注释（XML文件）的目录的路径，
            这些注释属于给定的各个图像目录中的图像。
            classes (list, optional): 包含对象类名称的列表，可以在“名称” XML标记中找到。
            include_classes (list, optional): “全部”或包含要包含在数据集中的类ID的整数列表。
            如果为“全部”，则所有真值框都将包含在数据集中。
            verbose (bool, optional): 如果为“ True”，则输出可能需要更长时间的操作进度。
        '''
        # 设置类成员。
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # 擦除以前可能已解析的数据。
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f]
                self.image_ids += image_ids

            if verbose:
                it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)),
                          file=sys.stdout)
            else:
                it = image_ids

            # 在数据集中循环所有的图片。
            for image_id in it:
                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    # 解析这个图片的XML文件。
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    boxes = []  # 存储这个照片的所有包围框。
                    objects = soup.find_all('object')  # 得到这张照片的所有对象列表。

                    # 解析每一个对象的数据。
                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        # 检查是否应将此类包含在数据集中。
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                        # 得到包围框的坐标。
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                        item_dict = {'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)

                    self.labels.append(boxes)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        return self.images, self.labels

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None
                 ):
        #label_encoder传入的是编码训练数据模块
        #############################################################################################
        # 对数据进行打乱
        #############################################################################################
        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        box_filter = BoxFilter(check_overlap=False,
                               check_min_area=False,
                               check_degenerate=True,
                               labels_format=self.labels_format)

        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # 生成最小批次
        #############################################################################################
        current = 0

        while True:
            batch_X, batch_y = [], []
            if current >= self.dataset_size:
                current = 0
                #########################################################################################
                # 数据循环完一轮后重新shuffle
                #########################################################################################
                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            batch_filenames = self.filenames[current:current + batch_size]
            for filename in batch_filenames:
                with Image.open(filename) as image:
                    batch_X.append(np.array(image, dtype=np.uint8))

            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current + batch_size])
            else:
                batch_y = None

            current += batch_size

            for i in range(len(batch_X)):
                if not (self.labels is None):
                    batch_y[i] = np.array(batch_y[i])

                # 应用我们得到的图片增广
                if transformations:
                    for transform in transformations:
                        if not (self.labels is None):
                            batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])
                        else:
                            batch_X[i] = transform(batch_X[i])

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or \
                            np.any(batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):  # 对错误的包围框进行过滤
                        batch_y[i] = box_filter(batch_y[i])

            batch_X = np.array(batch_X)
            #########################################################################################
            # 如果有标签编码器，编码我们的标签
            #########################################################################################

            if not (label_encoder is None or self.labels is None):
                batch_y_encoded = label_encoder(batch_y, diagnostics=False)
            else:
                batch_y_encoded = None

            ret = []
            ret.append(batch_X)
            ret.append(batch_y_encoded)

            yield tuple(ret)

    def get_dataset(self):
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        return self.dataset_size

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    Parser = DataGenerator(load_images_into_memory=True)
    # 包含图片文件夹。
    VOC_2007_images_dir = 'dataset/VOC2007/JPEGImages/'
    VOC_2012_images_dir = 'dataset/VOC2012/JPEGImages/'

    # 包含注解文件夹。
    VOC_2007_annotations_dir = 'dataset/VOC2007/Annotations/'
    VOC_2012_annotations_dir = 'dataset/VOC2012/Annotations/'

    # 图片集路径。
    VOC_2007_trainval_image_set_filename = 'dataset/VOC2007/ImageSets/trainval.txt'
    VOC_2012_trainval_image_set_filename = 'dataset/VOC2012/ImageSets/trainval.txt'

    images, labels = Parser.parse_xml(images_dirs=[VOC_2007_images_dir,
                                         VOC_2012_images_dir],
                            image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                                 VOC_2012_trainval_image_set_filename],
                            annotations_dirs=[VOC_2007_annotations_dir,
                                              VOC_2012_annotations_dir],
                            classes=classes
                            )

    for ix, image in enumerate(images):
        label = labels[ix]
        plt.imshow(image)
        for lab in label:
            rect = plt.Rectangle((lab[1], lab[2]), lab[3] - lab[1],
                                 lab[4] - lab[2], color='r', fill=False, linewidth=2)  # 左下起点，长，宽，颜色
            plt.text(lab[1], lab[2], classes[lab[0]], color='b')
            # 画矩形框
            plt.gca().add_patch(rect)
        plt.axis('off')  # 去掉坐标轴

        plt.show()