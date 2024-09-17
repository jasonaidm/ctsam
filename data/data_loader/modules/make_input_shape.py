import random
import cv2


class MakeInputShape(object):
    def __init__(self, height, width, method='padding', border_type='BORDER_CONSTANT'):
        self.height = height
        self.width = width
        self.method = method
        self.border_type = border_type if 'cv2.' in border_type \
            else 'cv2.'+border_type

    def __call__(self, data):
        if self.method != 'padding':
            raise NotImplementedError

        shape = data['img'].shape
        h, w = shape[:2]
        if (w / h) >= (self.width / self.height):
            data['img'] = cv2.resize(data['img'],
                                     dsize=(self.width, self.height),
                                     interpolation=cv2.INTER_LINEAR
                                     )
        else:
            new_w = int(w * self.height / h)
            data['img'] = cv2.resize(data['img'],
                                     dsize=(new_w, self.height),
                                     interpolation=cv2.INTER_LINEAR
                                     )
            delta_pixel = self.width - new_w
            right = random.randint(0, delta_pixel)
            left = delta_pixel - right
            if 'BORDER_CONSTANT' in self.border_type:
                if len(shape) == 3:
                    value = []
                    for i in range(shape[-1]):
                        value.append(int(data['img'][:, :, i].mean()))
                else:
                    value = int(data['img'].mean())
                data['img'] = cv2.copyMakeBorder(data['img'],
                                                 top=0,
                                                 bottom=0,
                                                 left=left,
                                                 right=right,
                                                 borderType=eval(self.border_type),
                                                 value=value
                                                 )
            else:
                data['img'] = cv2.copyMakeBorder(data['img'],
                                                 top=0,
                                                 bottom=0,
                                                 left=left,
                                                 right=right,
                                                 borderType=eval(self.border_type)
                                                 )
        return data


class MakeHeightShape(object):
    def __init__(self, height=32):
        self.height = height

    def __call__(self, data):
        shape = data['img'].shape
        h, w = shape[:2]
        scale = h / 32.0
        new_w = int(w / scale)
        data['img'] = cv2.resize(data['img'], dsize=(new_w, 32))
        return data


class ResizeCropPad(object):
    def __init__(self, fixed_size, deformation_fold):
        self.fixed_size = fixed_size
        self.deformation_fold = deformation_fold

    def __call__(self, data):
        img = data['img']
        h, w, _ = img.shape
        if h < w:
            if w < self.fixed_size:
                scale = self.fixed_size / w
                new_h = scale * h
                if self.deformation_fold * new_h < self.fixed_size:
                    top_p = (self.fixed_size - new_h) // 2
                    bottom_p = self.fixed_size - new_h - top_p
                    img = cv2.resize(img, (self.fixed_size, new_h))
                    img = cv2.copyMakeBorder(img, top_p, bottom_p, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
                else:
                    img = cv2.resize(img, (self.fixed_size, self.fixed_size))
            else:
                scale = self.fixed_size / h
                new_w = int(scale * w)
                if self.fixed_size * self.deformation_fold < new_w:
                    left_p = (new_w - self.fixed_size) // 2
                    # right_p = new_w - self.fixed_size - left_p
                    img = cv2.resize(img, (new_w, self.fixed_size))
                    img = img[:, left_p: left_p+self.fixed_size]
                else:
                    img = cv2.resize(img, (self.fixed_size, self.fixed_size))
        else:
            if h < self.fixed_size:
                scale = self.fixed_size / h
                new_w = scale * w
                if new_w * self.deformation_fold < self.fixed_size:
                    left_p = (self.fixed_size - new_w) // 2
                    right_p = self.fixed_size - new_w - left_p
                    img = cv2.resize(img, (new_w, self.fixed_size))
                    img = cv2.copyMakeBorder(img, 0, 0, left_p, right_p, cv2.BORDER_CONSTANT, value=(128, 128, 128))
                else:
                    img = cv2.resize(img, (self.fixed_size, self.fixed_size))
            else:
                scale = self.fixed_size / w
                new_h = int(scale * h)
                if self.fixed_size * self.deformation_fold < new_h:
                    top_p = (new_h - self.fixed_size) // 2
                    img = cv2.resize(img, (self.fixed_size, new_h))
                    img = img[top_p:top_p+self.fixed_size, :]
                else:
                    img = cv2.resize(img, (self.fixed_size, self.fixed_size))
        data['img'] = img
        return data


if __name__ == '__main__':
    mis = MakeInputShape(32, 280, border_type='BORDER_CONSTANT')
    img_path = r'D:\datasets\ocr_rec\baidu_rec\img\img_2708.jpg'
    img = cv2.imread(img_path, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = {'img': img}
    data = mis(data)
    print(data['img'].shape)
    cv2.imwrite('test.jpg', data['img'])
