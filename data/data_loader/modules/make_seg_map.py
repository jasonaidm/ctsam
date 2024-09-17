import numpy as np
import cv2
# from shapely.geometry import Polygon
# import pyclipper


def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class MakeSegMap(object):
    def __init__(self, min_polygon_size=8, multi_class=False):
        self.min_polygon_size = min_polygon_size
        self.multi_class = multi_class

    def __call__(self, data: dict) -> dict:
        image = data['img']
        seg_polys = data['eg_polys']
        ignore_tags = data.get('ignore_tags', {})
        h, w = image.shape[:2]
        seg_polys, ignore_tags = self.validate_polygons(seg_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(seg_polys)):
            polygon = seg_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_polygon_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                cv2.fillPoly(gt, [polygon.astype(np.int32)], 1)

        data['gt_map'] = gt
        data['mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            # if area > 0:
            #     polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        return cv2.contourArea(polygon)


if __name__ == '__main__':
    polygon = np.array([[0, 0], [100, 10], [100, 100], [10, 90]])
    a = shrink_polygon_py(polygon, 0.4)
    print(a)
    print(shrink_polygon_py(a, 1 / 0.4))
    b = shrink_polygon_pyclipper(polygon, 0.4)
    print(b)
    poly = Polygon(b)
    distance = poly.area * 1.5 / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(b, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    bounding_box = cv2.minAreaRect(expanded)
    points = cv2.boxPoints(bounding_box)
    print(points)
