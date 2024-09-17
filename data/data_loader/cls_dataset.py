import os
import json
import copy
import pandas as pd
import numpy as np
from data.data_loader.base_dataset import BaseDataset


class DCMFolderDataset(BaseDataset):
    def __init__(self, data_path, img_mode=None, pre_processes=None, transform=None, **kwargs):
        self.excel_path = kwargs.get('excel_path')
        self.num_valid_ctimgs = kwargs.get('num_valid_ctimgs', 64)
        self.num_classes = kwargs.get('num_classes', 2)
        if self.num_classes == 3:
            self.level_map = kwargs.get('level_map', {'PL0': 0, 'PL1': 1, 'PL2': 2})
        else:
            self.level_map = kwargs.get('level_map', {'PL0': 0, 'PL1': 1, 'PL2': 1})
        self.img_size = kwargs.get('img_size', 512)
        save_augimg = kwargs.get('save_augimg', {})
        sample_rate = kwargs.get('sample_rate', 1.0)
        super().__init__(data_path, img_mode, pre_processes, [], transform, sample_rate, save_augimg)
        
    
    def pre_load(self, data_path: str):
        data_list = []
        df = pd.read_excel(self.excel_path)
        id2label = {}
        for _, row in df.iterrows():
            id2label[row['owner_id']] = self.level_map[row['level']]
                
        for owner_folder in os.listdir(data_path):
            owner_id = int(owner_folder)
            if owner_id not in id2label:
                continue
            owner_folder_path = os.path.join(data_path, owner_folder)
            dcm_file_list = os.listdir(owner_folder_path)
            # 排序
            dcm_file_list.sort(key=lambda x: int(x.split('_')[0]))
            # ct片子数量
            if len(dcm_file_list) > self.num_valid_ctimgs:
                num_cut_off = (len(dcm_file_list) - self.num_valid_ctimgs)//2
                dcm_file_list = dcm_file_list[num_cut_off: -num_cut_off]
                
            dcm_file_path_list = [os.path.join(owner_folder_path, item) for item in dcm_file_list]
            
            data_list.append({
                'owner_id': owner_id,
                'dcm_file_path_list': dcm_file_path_list,
                'label': id2label[owner_id],
                'img_size': self.img_size
            })
        return data_list, len(data_list)

    def load_data(self, index):
        data = copy.deepcopy(self.data_handle[index])
        return data


class COCOJsonDataset(BaseDataset):
    def __init__(self, data_path, img_mode=None, pre_processes=None, transform=None, **kwargs):
        self.json_path = kwargs.get('json_path')
        self.excel_path = kwargs.get('excel_path')
        # self.dcm_folder = kwargs.get('dcm_folder')
        self.num_valid_ctimgs = kwargs.get('num_valid_ctimgs', 64)
        self.num_classes = kwargs.get('num_classes', 2)
        if self.num_classes == 3:
            self.level_map = kwargs.get('level_map', {'PL0': 0, 'PL1': 1, 'PL2': 2})
        else:
            self.level_map = kwargs.get('level_map', {'PL0': 0, 'PL1': 1, 'PL2': 1})
        self.img_size = kwargs.get('img_size', 512)
        save_augimg = kwargs.get('save_augimg', {})
        sample_rate = kwargs.get('sample_rate', 1.0)
        super().__init__(data_path, img_mode, pre_processes, [], transform, sample_rate, save_augimg)
        
    
    def pre_load(self, data_path: str):
        data_list = []
        df = pd.read_excel(self.excel_path)
        id2label = {}
        for _, row in df.iterrows():
            id2label[row['owner_id']] = self.level_map[row['level']]

        with open(self.json_path, encoding='utf-8') as f:
            instances_info = json.load(f)
        ann_dicts = instances_info['annotations']
        img_dicts = instances_info['images']
        img_id2bbox_info = {}
        for ann_dict in ann_dicts:
            if img_id2bbox_info.get(ann_dict['image_id']) is None:
                img_id2bbox_info[ann_dict['image_id']] = []
            img_id2bbox_info[ann_dict['image_id']].append({
                'bbox': ann_dict['bbox'],
                'category_id': ann_dict['category_id']}
            )

        img_df = pd.DataFrame(img_dicts)
        img_df['owner_id'] = img_df['file_name'].apply(lambda x: int(x.split('__')[0]))
        img_df['sequence'] = img_df['file_name'].apply(lambda x: int(x.split('__')[1].split('_')[0]))
        # 对onwer_id进行分组，然后分组排序
        img_groupby = img_df.groupby('owner_id')
        # 遍历
        for owner_id, sub_df in img_groupby:
            if owner_id not in id2label:  # 过滤掉没有label的owner_id
                continue
            # 对sub_df进行排序
            sub_df = sub_df.sort_values(['sequence'])
            # ct片子数量
            if len(sub_df) > self.num_valid_ctimgs:
                num_cut_off = (len(sub_df) - self.num_valid_ctimgs)//2
                sub_df = sub_df[num_cut_off: -num_cut_off]

            _data_list = []
            # 对sub_df进行遍历
            for _, row in sub_df.iterrows():
                _data_list.append({
                    'img_path': os.path.join(data_path, row['file_name']),
                    'bbox_info': img_id2bbox_info.get(row['id'], None),
                })
            data_list.append({
                'owner_id': owner_id,
                'img_data_info': _data_list,
                'label': id2label[owner_id],
                'img_size': self.img_size
            })
        
        return data_list, len(data_list)

    def load_data(self, index):
        data = copy.deepcopy(self.data_handle[index])
        return data


class DCMWithLabelDataset(BaseDataset):
    def __init__(self, data_path, img_mode, pre_processes, transform=None, **kwargs):
        self.dcm_folder = kwargs.get('dcm_folder')
        self.label_excel_file = kwargs.get('label_excel_file')
        save_augimg = kwargs.get('save_augimg', {})
        sample_rate = kwargs.get('sample_rate', 1.0)
        super().__init__(data_path, img_mode, pre_processes, [], transform, sample_rate, save_augimg)
        
    
    def pre_load(self, data_path: str):
        data_list = []
        with open(data_path, encoding='utf-8') as f:
            instances_info = json.load(f)
            
        df = pd.read_excel(self.label_excel_file)
        id2label = {}
        for _, row in df.iterrows():
            id2label[row['id']] = row['label']
        
        imgID2bboxes = {}
        for item in instances_info['annotations']:
            if imgID2bboxes.get(item['image_id']) is None:
                imgID2bboxes[item['image_id']] = [item['bbox']]
            else:
                imgID2bboxes[item['image_id']].append(item['bbox'])
        
        # 合并bboxes
        for imgID, bboxes in imgID2bboxes.items():
            imgID2bboxes[imgID] = self.merge_bboxes(bboxes) # TODO: dev merge_bboxes
        
        _dict = {}
        for item in instances_info['images']:
            img_name = item['file_name']
            owner_id = img_name.split('_')[0]
            dcm_name = img_name.replace(owner_id[0]+'_', '').replace('.png', '.dcm')
            # 去除'id' 前缀
            owner_id = owner_id.replace('id', '')
            dcm_file = os.path.join(self.dcm_folder, owner_id, dcm_name)
            
            if _dict.get('owner_id'):
                _dict['owner_id'].append(dcm_file)
            else:
                _dict['owner_id'] = [dcm_file]
        
        for owner_id, dcm_file_list in _dict.items():
            # 对dcm_name_list 进行升序
            dcm_file_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
            
            data_list.append({
                'owner_id': owner_id,
                'dcm_file_list': dcm_file_list,
                'label': id2label[owner_id],
            })
        return data_list, len(data_list)



class DCMWithCOCOLabelDataset(BaseDataset):
    def __init__(self, data_path, img_mode, pre_processes, transform=None, **kwargs):
        self.dcm_folder = kwargs.get('dcm_folder')
        self.img_folder = kwargs.get('img_folder')
        self.vol_size = kwargs.get('vol_size', 64)
        self.gt_vol_size = kwargs.get('gt_vol_size', 4) # 用于模型预测
        save_augimg = kwargs.get('save_augimg', {})
        sample_rate = kwargs.get('sample_rate', 1.0)
        filter_keys = kwargs.get('filter_keys', [])
        super().__init__(data_path, img_mode, pre_processes, filter_keys, transform, sample_rate, save_augimg)
        
    
    def pre_load(self, data_path: str):
        data_list = []
        with open(data_path, encoding='utf-8') as f:
            instances_info = json.load(f)
            
        imgID2Cocolabel = {}
        
        # 基于图片宽高信息，修正bbox
        imgID2wh = {}
        for item in instances_info['images']:
            imgID2wh[item['id']] = [item['width'], item['height']]

        for item in instances_info['annotations']:
            # 修正bbox
            bbox = item['bbox'] 
            bbox[0] *= 512 / imgID2wh[item['image_id']][0] 
            bbox[1] *= 512 / imgID2wh[item['image_id']][1]
            bbox[2] *= 512 / imgID2wh[item['image_id']][0]
            bbox[3] *= 512 / imgID2wh[item['image_id']][1]
            label_dict = {'bbox': bbox, 'category_id': item['category_id']}
            if imgID2Cocolabel.get(item['image_id']) is None:
                imgID2Cocolabel[item['image_id']] = [label_dict]
            else:
                tmp_bbox = imgID2Cocolabel[item['image_id']][0]['bbox']  # 只考虑最偏右下角的bbox
                if (tmp_bbox[0]-item['bbox'][0]) + (tmp_bbox[1]-item['bbox'][1]) < 0:
                    imgID2Cocolabel[item['image_id']][0] = label_dict

                # imgID2Cocolabel[item['image_id']].append(label_dict)
        
        # 聚合ct扫面图片，得到volume
        _dict = {}
        dcmFile2Label = {}
        for item in instances_info['images']:
            img_name = item['file_name']
            img_id = item['id']
            owner_id, jpg_name = img_name.split('__')
            if self.dcm_folder:
                dcm_name = jpg_name.replace('.jpg', '.dcm')
                dcm_file = os.path.join(self.dcm_folder, owner_id, dcm_name)
            else:
                dcm_file = os.path.join(self.img_folder, img_name)
            dcmFile2Label[dcm_file] = imgID2Cocolabel.get(img_id, None)
            
            if _dict.get(owner_id):
                _dict[owner_id].append(dcm_file)
            else:
                _dict[owner_id] = [dcm_file]
        
        for owner_id, dcm_file_list in _dict.items():
            # 对dcm_name_list 进行升序
            dcm_file_list.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
            # volume size
            if len(dcm_file_list) > self.vol_size:
                start_idx = (len(dcm_file_list) - self.vol_size) // 2
                end_idx = start_idx + self.vol_size
                dcm_file_list = dcm_file_list[start_idx: end_idx]

            label_list = [dcmFile2Label[x] for x in dcm_file_list]
            # 获取裁剪坐标x,y
            gt_start_idx = (len(label_list) - self.gt_vol_size) // 2
            gt_end_idx = gt_start_idx + self.gt_vol_size
            gt_label_list = label_list[gt_start_idx: gt_end_idx]
            center_x = []
            center_y = []
            for idx, item in enumerate(gt_label_list):
                if item is None:   # TODO: 考虑ct扫面图片中没有标注的情况，新增一个空bbox，类别为0，训练时，只计算其是否为前景的置信度loss
                    # center_x.append(256)
                    # center_y.append(256)
                    # gt_label_list[idx] = [{'bbox':[256, 256, 0, 0], 'category_id': 3}]
                    pass
                else:
                    center_x.append(item[0]['bbox'][0] + item[0]['bbox'][2]/2)
                    center_y.append(item[0]['bbox'][1] + item[0]['bbox'][3]/2)
            # 标记图少于4的样本，不参与训练
            if len(center_x) < 4:
                continue
            
            if len(center_x) > 4:
                print(center_x)
                
            data_list.append({
                'owner_id': owner_id,
                'dcm_file_list': dcm_file_list,
                'label_list': label_list,
                'center_x': int(np.mean(center_x)),
                'center_y': int(np.mean(center_y)),
                'gt_label_list': gt_label_list
            })
        # print("##有效样本数：{}".format(len(data_list)))
        return data_list, len(data_list)

    def load_data(self, index):
        data = copy.deepcopy(self.data_handle[index])
        return data
