import cv2
import sys
import os
import numpy as np
import pydicom
from tqdm import tqdm
import pandas as pd
import glob
import pdb


# 加载 Dicom图像
def get_pixeldata(dicom_dataset):
    if dicom_dataset.BitsAllocated == 1:
        # single bits are used for representation of binary data
        format_str = 'uint8'
    elif dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    numpy_dtype = np.dtype(format_str)


    if dicom_dataset.is_little_endian != (sys.byteorder == 'little'):
        numpy_dtype = numpy_dtype.newbyteorder('S')

    pixel_bytearray = dicom_dataset.PixelData

    if dicom_dataset.BitsAllocated == 1:
        # if single bits are used for binary representation, a uint8 array
        # has to be converted to a binary-valued array (that is 8 times bigger)
        try:
            pixel_array = np.unpackbits(
                np.frombuffer(pixel_bytearray, dtype='uint8'))
        except NotImplementedError:
            # PyPy2 does not implement numpy.unpackbits
            raise NotImplementedError(
                'Cannot handle BitsAllocated == 1 on this platform')
    else:
        pixel_array = np.frombuffer(pixel_bytearray, dtype=numpy_dtype)
    length_of_pixel_array = pixel_array.nbytes
    expected_length = dicom_dataset.Rows * dicom_dataset.Columns
    if ('NumberOfFrames' in dicom_dataset and
            dicom_dataset.NumberOfFrames > 1):
        expected_length *= dicom_dataset.NumberOfFrames
    if ('SamplesPerPixel' in dicom_dataset and
            dicom_dataset.SamplesPerPixel > 1):
        expected_length *= dicom_dataset.SamplesPerPixel
    if dicom_dataset.BitsAllocated > 8:
        expected_length *= (dicom_dataset.BitsAllocated // 8)
    padded_length = expected_length
    if expected_length & 1:
        padded_length += 1
    if length_of_pixel_array != padded_length:
        raise AttributeError(
            "Amount of pixel data %d does not "
            "match the expected data %d" %
            (length_of_pixel_array, padded_length))
    if expected_length != padded_length:
        pixel_array = pixel_array[:expected_length]

    if dicom_dataset.Modality.lower().find('ct') >= 0:  # CT图像需要得到其CT值图像
        pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept  # 获得图像的CT值
    pixel_array = pixel_array.reshape(dicom_dataset.Rows, dicom_dataset.Columns*dicom_dataset.SamplesPerPixel)
    
    return pixel_array


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    """设置CT图像的窗宽和窗位"""
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    img_array = np.clip(img_temp, 0, 255)

    return img_array


class ConvertClass(object):
    def __init__(self, dcm, winwidth=None, wincenter=None):
        super(ConvertClass, self).__init__()

        self.pixel_array = dcm.pixel_array
        
        if winwidth is not None and wincenter is not None:
            self.wc = wincenter
            self.ww = winwidth
        else:
            if type(dcm.WindowCenter).__name__ == 'DSfloat':
                self.wc = int(dcm.WindowCenter)
                self.ww = int(dcm.WindowWidth)
            elif type(dcm.WindowCenter).__name__ == 'MultiValue':
                self.wc = int(dcm.WindowCenter[0])
                self.ww = int(dcm.WindowWidth[0])
            
        self.ymax = 255
        self.ymin = 0

        if hasattr(dcm, 'RescaleSlope'):
            self.slope = dcm.RescaleSlope
        else:
            self.slope = 1
        if hasattr(dcm, 'RescaleIntercept'):
            self.intercept = dcm.RescaleIntercept
        else:
            self.intercept = 0

        if hasattr(dcm, 'PhotometricInterpretation'):
            self.PhotometricInterpretation = dcm.PhotometricInterpretation
        else:
            self.PhotometricInterpretation = 'other'

        if hasattr(dcm, 'VOILUTFunction'):
            self.VOILUTFunction = dcm.VOILUTFunction
        else:
            self.VOILUTFunction = 'LINEAR'

    # VOILUTFunction linear_exact
    def linear_exact(self):
        pixel_array = self.pixel_array * self.slope + self.intercept
        linear_exact_array_less_idx = pixel_array <= (self.wc - self.ww / 2)
        linear_exact_array_large_idx = pixel_array > (self.wc + self.ww / 2)
        linear_exact_array = ((self.pixel_array - self.wc) / self.ww + 0.5) * (self.ymax - self.ymin) + self.ymin

        linear_exact_array[linear_exact_array_less_idx] = self.ymin
        linear_exact_array[linear_exact_array_large_idx] = self.ymax
        linear_exact_array = linear_exact_array.astype(np.uint8)

        if self.PhotometricInterpretation == 'MONOCHROME1':
            linear_exact_array = 255 - linear_exact_array
        return linear_exact_array

    # VOILUTFunction sigmoid
    def sigmoid(self):
        pixel_array = self.pixel_array * self.slope + self.intercept
        sigmoid_array = (self.ymax - self.ymin)/(1+np.exp(-4*(pixel_array-self.wc)/self.ww)) + self.ymin
        sigmoid_array = sigmoid_array.astype(np.uint8)

        if self.PhotometricInterpretation == 'MONOCHROME1':
            sigmoid_array = 255 - sigmoid_array
        return sigmoid_array

    # VOILUTFunction linear
    def linear(self):
        pixel_array = self.pixel_array * self.slope + self.intercept
        linear_array_less_idx = pixel_array <= (self.wc - self.ww/2)
        linear_array_large_idx = pixel_array > (self.wc + self.ww/2 - 1)
        linear_array = ((pixel_array - (self.wc - 0.5))/(self.ww - 1) + 0.5) * (self.ymax - self.ymin) + self.ymin

        linear_array[linear_array_less_idx] = self.ymin
        linear_array[linear_array_large_idx] = self.ymax
        linear_array = linear_array.astype(np.uint8)

        if self.PhotometricInterpretation == 'MONOCHROME1':
            linear_array = 255 - linear_array

        return linear_array

    def noWW(self):
        min_value = np.min(self.pixel_array)
        max_value = np.max(self.pixel_array)
        pixel_array = (self.pixel_array - min_value) / (max_value - min_value) * 255
        pixel_array = pixel_array.astype(np.uint8)

        if self.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = 255 - pixel_array
        return pixel_array
    

def dcm2img_by_method1(dcm_path, winwidth, wincenter, save_path=None):
    """将dcm文件转换为png文件"""
    dcm = pydicom.read_file(dcm_path)
    pixel_array = get_pixeldata(dcm)
    img_array = setDicomWinWidthWinCenter(pixel_array, winwidth, wincenter, dcm.Rows, dcm.Columns)
    if save_path:
        cv2.imwrite(save_path, img_array)
    return img_array


def dcm2img_by_method2(dcm_path, winwidth, wincenter, save_path=None, proc_method='linear'):
    dcm = pydicom.read_file(dcm_path)
    convert = ConvertClass(dcm, winwidth, wincenter)
    if proc_method == 'sigmoid':
        img_array = convert.sigmoid()
    
    elif proc_method == 'linear':
        img_array = convert.linear()
    
    elif proc_method == 'noWW':
        img_array = convert.noWW()

    elif proc_method == 'linear_exact':
        img_array = convert.linear_exact()
    if save_path:
        cv2.imwrite(save_path, img_array)
    return img_array


def test_diff_method(winwidth, wincenter, dcm_dir, out_dir1, out_dir2):
    from time import time
    timer1 = 0
    timer2 = 0
    for idx, dcm_file in enumerate(os.listdir(dcm_dir)):
        dcm_path = os.path.join(dcm_dir, dcm_file)
        save_path1 = os.path.join(out_dir1, dcm_file.replace('dcm', 'jpg'))
        save_path2 = os.path.join(out_dir2, dcm_file.replace('dcm', 'jpg'))
        start_time = time()
        dcm2img_by_method1(dcm_path, winwidth, wincenter, save_path1)
        timer1 += time() - start_time
        start_time = time()
        dcm2img_by_method2(dcm_path, winwidth, wincenter, save_path2)
        timer2 += time() - start_time
    
    print('## method1 elapsed time:', timer1/(idx+1))
    print('## method2 elapsed time:', timer2/(idx+1))


def dcm2img_by_dir0(dcm_dir, out_dir, match_mode='*/*.dcm', winwidth=1600, wincenter=-600, extract_num=None, proc_method='linear'):
    os.makedirs(out_dir, exist_ok=True)
    dcm_list = glob.glob(os.path.join(dcm_dir, match_mode))
    total_num = len(dcm_list)
    if extract_num:
        start_idx = (total_num-extract_num) // 2 + 1
        end_idx = start_idx + extract_num - 1
    else:
        start_idx = 1
        end_idx = total_num
    for dcm_path in tqdm(dcm_list):
        parent_dir, dcm_file = os.path.split(dcm_path)
        if len(dcm_file.split('.')) != 2:
            pdb.set_trace()
            raise ValueError('dcm_file should be like xx_xx.dcm')
        
        prefix=parent_dir.replace(dcm_dir, '').strip('/').strip('\\')
        seq_num = int(dcm_file.split('_')[0])
        if start_idx <= seq_num <= end_idx:
            img_path = os.path.join(out_dir, prefix + '__' + dcm_file.replace('.dcm', '') + '.jpg')
            dcm2img_by_method2(dcm_path, winwidth, wincenter, save_path=img_path, proc_method=proc_method)
            
    return out_dir


def dcm2img_by_dir(dcm_dir, out_dir, match_mode='*/*/*.dcm', winwidth=1600, wincenter=-600, extract_num=None, proc_method='linear'):
    os.makedirs(out_dir, exist_ok=True)
    dcm_list = glob.glob(os.path.join(dcm_dir, match_mode))
    total_num = len(dcm_list)
    if extract_num:
        start_idx = (total_num-extract_num) // 2 + 1
        end_idx = start_idx + extract_num - 1
    else:
        start_idx = 1
        end_idx = total_num
    for dcm_path in tqdm(dcm_list):
        parent_dir, dcm_file = os.path.split(dcm_path)
        if len(dcm_file.split('.')) != 2:
            pdb.set_trace()
            raise ValueError('dcm_file should be like xx_xx.dcm')

        seq_num = int(dcm_file.split('_')[0])
        if start_idx <= seq_num <= end_idx:
            prefix=parent_dir.replace(dcm_dir, '').strip('/').split('/')[0]
            img_path = os.path.join(out_dir, prefix + '__' + dcm_file.replace('.dcm', '') + '.jpg')
            dcm2img_by_method2(dcm_path, winwidth, wincenter, save_path=img_path, proc_method=proc_method)
            
    return out_dir


def dcm2img_by_dir2(dcm_dir, out_dir, excel_path, match_mode='*/*/*.dcm', 
                    winwidth=1600, wincenter=-600, 
                    extract_num=64, 
                    proc_method='linear'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['id'])
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        center_idx = row['index']
        if not center_idx:
            continue
        center_idx = str(center_idx).split('；')[0]
        # 过滤掉非数字
        if not center_idx.isdigit():
            continue
        center_idx = int(center_idx)
        
        obj_id = str(int(row['id']))
        obj_dir = os.path.join(dcm_dir, obj_id)
        if not os.path.exists(obj_dir):
            continue
        # if len(os.listdir(obj_dir)) != 1:
        #     print(f"## warning: The data format of idx_{obj_id} look an anomaly!")
        #     continue

        dcm_list = glob.glob(os.path.join(obj_dir, match_mode))
        start_idx = center_idx - extract_num//2
        end_idx = center_idx + extract_num//2

        for dcm_path in dcm_list:
            parent_dir, dcm_file = os.path.split(dcm_path)
            if len(dcm_file.split('.')) != 2:
                pdb.set_trace()
                raise ValueError('dcm_file should be like xx_xx.dcm')

            seq_num = int(dcm_file.split('_')[0])
            if start_idx <= seq_num <= end_idx:
                # prefix=parent_dir.replace(dcm_dir, '').strip('/').split('/')[0]
                img_path = os.path.join(out_dir, obj_id + '__' + dcm_file.replace('.dcm', '') + '.jpg')
                dcm2img_by_method2(dcm_path, winwidth, wincenter, save_path=img_path, proc_method=proc_method)


if __name__ == '__main__':
    winwidth = 1600
    wincenter = -600
    dcm_dir = '/mnt/f/medical_ct_data/final_data/pos'
    out_dir = '/mnt/f/medical_ct_data/images/pos'
    excel_path = '/mnt/e/ai_dev/ct_model/data/raw_data/pos_center_index.xlsx'
    dcm2img_by_dir2(dcm_dir, out_dir, excel_path, match_mode='*/*.dcm')
