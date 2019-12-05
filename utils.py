import numpy as np

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    在两种坐标格式之间转换轴对齐2D框的坐标。
     下面有3中模式可以互相转换：
        1) (xmin, xmax, ymin, ymax) - 'minmax' 格式
        2) (xmin, ymin, xmax, ymax) - 'corners' 格式
        2) (cx, cy, w, h) - 'centroids' 格式
    Arguments:
        tensor (array): 一个Numpy nD数组，其中包含要在最后一个轴上的某个位置转换的四个连续坐标。
        start_index (int): 张量的最后一个轴上的第一个坐标的索引。
        conversion (str, optional): 转换方式。
        border_pixels (str, optional): 如何处理边界框的边框像素。
    Returns:
        一个Numpy nD数组，输入张量的副本，其中转换后的坐标代替原始坐标，而原始张量的未更改元素在其他位置。
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

def intersection_area_(boxes1, boxes2,  mode='outer_product'):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if mode == 'outer_product':

        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, 0] * side_lengths[:, 1]

def iou(boxes1, boxes2, mode='outer_product', border_pixels='half'):
    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
    boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')

    intersection_areas = intersection_area_(boxes1, boxes2, mode=mode)

    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':

        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas

class BoundGenerator:   #生成裁剪的区域范围
    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError("All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None: bound_pair[0] = 0.0   #如果第一个值为None的话将其置为0.0
            if bound_pair[1] is None: bound_pair[1] = 1.0   #如果第二个值为None的话将其置为1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError("For all sample space elements, the lower bound cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:  #将值进行缩放
            self.weights = [1.0/self.sample_space_size] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]   #随机返回一种方式

class BoxFilter:
    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # 记录通过检测的所有包围框。
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:
            non_degenerate = (labels[:, xmax] > labels[:, xmin]) * (labels[:, ymax] > labels[:, ymin])
            requirements_met *= non_degenerate  #计算非空的包围框

        if self.check_min_area:
            min_area_met = (labels[:, xmax] - labels[:, xmin]) * (labels[:, ymax] - labels[:, ymin]) >= self.min_area
            requirements_met *= min_area_met   #选取比最小面积大的包围框

        if self.check_overlap:
            # 获取上下限。
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # 计算无效的包围框。
            if self.overlap_criterion == 'iou':
                # 计算面片坐标。
                image_coords = np.array([0, 0, image_width, image_height])
                # 计算补丁和所有地面真相框之间的IoU。
                image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]], mode='element-wise', border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou > lower) * (image_boxes_iou <= upper)  #过滤掉iou不在范围内的包围框

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    d = 1
                elif self.border_pixels == 'exclude':
                    d = -1
                # 计算包围框的面积
                box_areas = (labels[:, xmax] - labels[:, xmin] + d) * (labels[:, ymax] - labels[:, ymin] + d)
                # 计算填补部分和所有真实包围框的交集
                clipped_boxes = np.copy(labels)
                clipped_boxes[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=image_height-1)
                clipped_boxes[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=image_width-1)
                intersection_areas = (clipped_boxes[:,xmax] - clipped_boxes[:,xmin] + d) * \
                                     (clipped_boxes[:,ymax] - clipped_boxes[:,ymin] + d)
                # 检查那些包围框符合重叠要求
                if lower == 0.0:
                    mask_lower = intersection_areas > lower * box_areas
                else:
                    mask_lower = intersection_areas >= lower * box_areas
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # 计算包围框的中心点
                cy = (labels[:, ymin] + labels[:, ymax]) / 2
                cx = (labels[:, xmin] + labels[:, xmax]) / 2
                # 移除不符合要求的包围框
                requirements_met *= (cy >= 0.0) * (cy <= image_height-1) * (cx >= 0.0) * (cx <= image_width-1)

        return labels[requirements_met]