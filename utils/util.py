# -*- coding: utf-8 -*-
import json
import pathlib
import time
import os
import math
import glob
import cv2
import numpy as np
import pandas as pd
import anyconfig
from PIL import Image, ImageDraw, ImageFont
import base64


OPENCV_VERSION = int(cv2.__version__.split('.')[0])


def get_df(label_file, *args, **kwargs):
    with open(label_file, encoding='utf-8') as f:
        instances_info = json.load(f)  # 目前默认所以json里的categories相同

    # 类别dict
    category_dict = {item['id']:item['name'] for item in instances_info['categories']}
    # cate_df = pd.DataFrame(instances_info['categories'])
    img_df = pd.DataFrame(instances_info['images'])
    ann_df = pd.DataFrame(instances_info['annotations'])
    img_df.rename(columns={'id': 'image_id'}, inplace=True)
    merge_df = pd.merge(img_df, ann_df, on='image_id', how='outer')
    return merge_df, category_dict, instances_info


def bbox_iou2(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
  
    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1) 
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算交并比（交集/并集）
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def bbox_iou(bbox1, bbox2):
    xmin1, ymin1, w1, h1 = bbox1
    xmin2, ymin2, w2, h2 = bbox2
    xmax1 = xmin1 + w1
    xmax2 = xmin2 + w2
    ymax1 = ymin1 + h1
    ymax2 = ymin2 + h2
  
    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1) 
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算交并比（交集/并集）
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def save_img3d(img3d, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(img3d.shape[0]):
        img = img3d[i].astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'{i}.jpeg'), img)
    return


def save_bbox_imgs(img_dir, bbox_img_dir, file_name, bbox, idx, label, pred):
    os.makedirs(bbox_img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, file_name)
    tag = file_name.split('.')[0]
    new_fname = f"{tag}_{label}_{pred}_{idx}.jpeg"
    bbox_img_path = os.path.join(bbox_img_dir, new_fname)
    img = cv2.imread(img_path, flags=0)
    height, width = img.shape
    bbox = [int(i) for i in bbox]
    x1, y1, w, h = bbox
    # 扩大边缘
    expand_rate = 1.0
    max_expand = 256
    delta_x = int(min(w*expand_rate, max_expand))
    delta_y = int(min(h*expand_rate, max_expand))
    x2 = x1+w
    y2 = y1+h
    x1 -= delta_x
    y1 -= delta_y
    x2 += delta_x
    y2 += delta_y

    # clip
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    # w = x2 - x1
    # h = y2 - y1
        # 画框
    x, y, w, h = bbox
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
    bbox_img = img[y1: y2, x1: x2].copy()
    cv2.imwrite(bbox_img_path, bbox_img)
    return


def setup_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('REC.DEV')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """
    将一个list的数组写入txt文件里
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_img(imgs: np.ndarray, title='img'):
    import matplotlib.pyplot as plt
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    i = 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA)
        i += 1
    return tmp


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    # import pdb;pdb.set_trace()
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data


def replace_rare_word(rare_word_path):
    map_dict = {}
    with open(rare_word_path, encoding='utf-8') as f:
        for i in f:
            line = i.rstrip('\n').split('\t')
            if len(line) == 3:
                map_dict[line[1]] = line[-1]
    return map_dict


def parse_config(config: dict) -> dict:
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def save_result(result_path, box_list, score_list, is_output_polygon):
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    """
    对只有一个字符的框进行扩充
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def find_contours(mask, mode=cv2.RETR_CCOMP, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    if OPENCV_VERSION < 4:
        _, contours, hierarchy = cv2.findContours(mask, mode=mode, method=method)
    else:
        contours, hierarchy = cv2.findContours(mask, mode=mode, method=method)
    return contours, hierarchy


def get_max_contour_vertex(img, approx_eps=0.02, denoise=False):
    """获取最大轮廓四边形区域的4个顶点，可用于定位table roi"""
    # 转化为二值图
    if denoise:
        thresh = cv2.adaptiveThreshold(~img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -10)
        Structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, Structure, (-1, -1))
        thresh = cv2.dilate(thresh, Structure, (-1, -1))
        img = thresh
        # cv2.imwrite('debug/table_thresh.jpg', thresh)
    # 提取轮廓
    contours, hierarchy = find_contours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 逼近多边形
    max_area = 20
    max_idx = 0
    img_copy = img.copy()
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        if area > max_area:
            max_area = area
            max_idx = idx

        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 255), 2)
    # 近似四边形逼近
    approx = cv2.approxPolyDP(contours[max_idx], approx_eps * cv2.arcLength(contours[max_idx], True), True)
    # 检测该轮廓的凸包
    hull = cv2.convexHull(approx)

    return hull  # 返回表格的4个顶点


def getfourpoints(hull: list, value=12, boder=0):
    """
    获取四边形的左上、右上、右下、左下这4个顶点
    hull=cv2.convexHull(approx).tolist()
    """
    minlist = sorted(hull)
    # 左上角坐标
    left = (minlist[0][0][0] + boder, minlist[0][0][1] + boder)
    # 左下角坐标
    leftbottom = []
    for i, p in enumerate(minlist):
        if i > 0:
            if p[0][1] - minlist[0][0][1] > value:
                leftbottom = (p[0][0] + boder, p[0][1] + boder)
                break
            elif p[0][1] - minlist[0][0][1] < -value:
                leftbottom = left
                left = (p[0][0] + boder, p[0][1] + boder)
                break
            else:
                continue
    # 右下角坐标
    data = np.array(hull)
    flist = data[np.lexsort(-data.T)]
    # 第一行的差值
    wjh = abs(flist[0][0][0][0] - flist[0][0][0][1])
    # 第二行的差值
    pwjh = abs(flist[0][1][0][0] - flist[0][1][0][1])
    # 第一行宽减第二行宽
    ojtw = flist[0][0][0][0] - flist[0][1][0][0]
    # 第一行高减第二行高
    ojth = flist[0][0][0][1] - flist[0][1][0][1]
    if ojtw < 0 and pwjh <= wjh:
        rightbottom = (flist[0][1][0][0] + boder, flist[0][1][0][1] + boder)
    else:
        rightbottom = (flist[0][0][0][0] + boder, flist[0][0][0][1] + boder)

    zlist = data[np.lexsort(data.T)][0]
    # 右上角坐标
    righttop = []
    for i, p in enumerate(zlist):
        if i > 0:
            if p[0][0] - left[0] > value:
                righttop = (p[0][0] + boder, p[0][1] + boder)
                break
            else:
                continue
        else:
            if p[0][0] - left[0] > value:
                righttop = (p[0][0] + boder, p[0][1] + boder)
                break
    vertex = [left, righttop, rightbottom, rightbottom]
    # 如果存在空则未找全4个角坐标
    if len(left) == 0 or len(leftbottom) == 0 or len(rightbottom) == 0 or len(righttop) == 0:
        vertex = []
    return vertex


def get_line_intersection(line1, line2):
    # 求线段(延长线)和直线交点
    k1, b1 = line1
    k2, b2 = line2
    x = (b1 - b2) / (k2 - k1)
    y = ((k1 + k2)*x + b1 + b2) * 0.5

    return round(x), round(y)


def get_segment_intersection(line1, line2):
    # 求两直线交点
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    if x1 == x2:  # L1直线斜率不存在
        y = (y4-y3) * (x1-y4) / (x4-x3) + y4
        return round(x1), round(y)
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return round(x), round(y)


def base2mat(img_base64):
    imgData = base64.b64decode(img_base64)
    nparr = np.fromstring(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def mat2base(img_np):
    image = cv2.imencode('.png', img_np)[1]
    img_base64 = str(base64.b64encode(image))[2:-1]
    return img_base64


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif'}
    if os.path.isfile(img_file) and os.path.splitext(img_file)[-1][1:].lower(
    ) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[-1][
                    1:].lower() in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def bb_intersection_over_union(boxA, boxB):
    # boxA = [int(x) for x in boxA]
    # boxB = [int(x) for x in boxB]
    if (boxB[0] >= boxA[2]) or (boxB[1] >= boxA[3]) or (boxB[2] <= boxA[0]) or (boxB[3] <= boxA[1]):
        return 0.0, 0.0, 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxAWidth = (boxA[2] - boxA[0] + 1)
    boxAHeight = (boxA[3] - boxA[1] + 1)
    iou_area = interArea / float(boxAArea)  # 非标准IOU，交集占检测框百分比
    iou_x = interWidth / boxAWidth
    iou_y = interHeight / float(boxAHeight)
    return iou_area, iou_x, iou_y


def sorted_boxes_bak(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def sorted_boxes(dt_boxes, x_shift_weight=0.3, y_shift_weight=0.6):
    """
    文本框自上而下，自左而右排序
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    # _boxes = list(sorted_boxes)

    # 计算可对比的两个空间指标
    _boxes = []
    for i in range(num_boxes):
        x_flag = sorted_boxes[i][0][0] * x_shift_weight + sorted_boxes[i][2][0] * (1 - x_shift_weight)
        y_flag = sorted_boxes[i][0][1] * y_shift_weight + sorted_boxes[i][3][1] * (1 - y_shift_weight)
        _boxes.append([sorted_boxes[i][0][0], x_flag, sorted_boxes[i][3][1], y_flag, i])

    replace_count = 0
    while True:
        replace_instruct = False
        for i in range(num_boxes - 1):
            if (_boxes[i + 1][3] < _boxes[i][2]) and (_boxes[i + 1][1] < _boxes[i][0]):  # and (_boxes[i + 1][2] > _boxes[i][3])
                replace_instruct = True
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp

        if replace_instruct:
            replace_count += 1
        else:
            break
    # print("## 文本框排序迭代次数： %d" % replace_count)
    new_sorted_boxes = [sorted_boxes[i[-1]] for i in _boxes]

    return new_sorted_boxes


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.65), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def remove_blank(char_topk, prob_topk, is_remove_duplicate=True, blank_symbol=''):
    """移除ctc输出的blank"""
    char_list = []
    prob_list = []
    for i in range(len(char_topk)):
        if char_topk[i][0] != blank_symbol:
            if is_remove_duplicate and (i > 0 and char_topk[i - 1][0] == char_topk[i][0]):
                continue
            char_list.append(char_topk[i])
            prob_list.append(prob_topk[i])
    return char_list, prob_list


def draw_bbox_region(row, idx, img_path, label_dir):
    cls_id = row['category_id']
    bbox_id = row.get('id', idx)
    fname = os.path.basename(img_path).split('.')[0]
    new_fname = f"{fname}_{cls_id}_{bbox_id}.jpeg"
    bbox_img_path = os.path.join(label_dir, new_fname)
    img = cv2.imread(img_path, flags=0)
    height, width = img.shape
    x1, y1, w, h = [int(i) for i in row['bbox']]
    # 扩大边缘
    expand_rate = 1.0
    max_expand = 256
    delta_x = int(min(w*expand_rate, max_expand))
    delta_y = int(min(h*expand_rate, max_expand))
    x2 = x1+w
    y2 = y1+h
    x1 -= delta_x
    y1 -= delta_y
    x2 += delta_x
    y2 += delta_y

    # clip
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    # w = x2 - x1
    # h = y2 - y1
        # 画框
    x, y, w, h = [int(i) for i in row['bbox']]
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
    bbox_img = img[y1: y2, x1: x2].copy()
    
    cv2.imwrite(bbox_img_path, bbox_img)

