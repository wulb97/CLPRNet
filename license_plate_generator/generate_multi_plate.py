import numpy as np
import cv2, os, argparse
from glob import glob
from license_plate_generator.plate_number import letters, digits
import random


def get_location_data(length=7, split_id=1, height=140):
    """
    获取车牌号码在底牌中的位置
    length: 车牌字符数，7或者8，7为普通车牌、8为新能源车牌
    split_id: 分割空隙
    height: 车牌高度，对应单层和双层车牌
    """
    # 字符位置
    location_xy = np.zeros((length, 4), dtype=np.int32)

    # 单层车牌高度
    if height == 140:
        # 单层车牌，y轴坐标固定
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115
        # 螺栓间隔
        step_split = 34 if length == 7 else 49
        # 字符间隔
        step_font = 12 if length == 7 else 9

        # 字符宽度
        width_font = 45
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            # 新能源车牌
            if length == 8 and i > 0:
                width_font = 43
            location_xy[i, 2] = location_xy[i, 0] + width_font
    else:
        # 双层车牌第一层
        location_xy[0, :] = [110, 15, 190, 75]
        location_xy[1, :] = [250, 15, 330, 75]

        # 第二层
        width_font = 65
        step_font = 15
        for i in range(2, length):
            location_xy[i, 1] = 90
            location_xy[i, 3] = 200
            if i == 2:
                location_xy[i, 0] = 27
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    return location_xy


# 字符贴上底板
def copy_to_image_multi(img, font_img, bbox, bg_color, is_red):
    x1, y1, x2, y2 = bbox
    font_img = cv2.resize(font_img, (x2 - x1, y2 - y1))
    img_crop = img[y1: y2, x1: x2, :]

    if is_red:
        img_crop[font_img < 200, :] = [0, 0, 255]
    elif 'blue' in bg_color or 'black' in bg_color:
        img_crop[font_img < 200, :] = [255, 255, 255]
    else:
        img_crop[font_img < 200, :] = [0, 0, 0]
    return img

def sp_noise(image, prob):
    """添加椒盐噪声prob:噪声比例"""
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gasuss_noise(image, mean=0, var=0.001):
    """添加高斯噪声mean : 均值var : 方差"""
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

class MultiPlateGenerator:
    def __init__(self, adr_plate_model, adr_font):
        # 车牌底板路径
        self.adr_plate_model = adr_plate_model
        # 车牌字符路径
        self.adr_font = adr_font

        # 车牌字符图片，预存处理
        self.font_imgs = {}
        font_filenames = glob(os.path.join(adr_font, '*jpg'))
        for font_filename in font_filenames:
            # font_img = cv2.imread(font_filename, cv2.IMREAD_GRAYSCALE)
            font_img = cv2.imdecode(np.fromfile(font_filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if '140' in font_filename:
                font_img = cv2.resize(font_img, (45, 90))
            elif '220' in font_filename:
                font_img = cv2.resize(font_img, (65, 110))
            elif font_filename.split('_')[-1].split('.')[0] in letters + digits:
                font_img = cv2.resize(font_img, (43, 90))
            self.font_imgs[os.path.basename(font_filename).split('.')[0]] = font_img

        # 字符位置
        self.location_xys = {}
        for i in [7, 8]:
            for j in [1, 2, 3, 4]:
                for k in [140, 220]:
                    self.location_xys['{}_{}_{}'.format(i, j, k)] = \
                        get_location_data(length=i, split_id=j, height=k)

    # 获取字符位置
    def get_location_multi(self, plate_number, height=140):
        length = len(plate_number)
        if '警' in plate_number:
            split_id = 1
        elif '使' in plate_number:
            split_id = 3
        elif '领' in plate_number:
            split_id = 4
        else:
            split_id = 2
        return self.location_xys['{}_{}_{}'.format(length, split_id, height)]

    
    def generate_plate_special(self, plate_number, bg_color, is_double, enhance=False):
        """
        生成特定号码、颜色车牌
        :param plate_number: 车牌号码
        :param bg_color: 背景颜色，black/black_shi（使领馆）/blue/green_car（新能源轿车）/green_truck（新能源卡车）/white/white_army（军队）/yellow
        :param is_double: 是否双层
        :param enhance: 图像增强
        :return: 车牌图
        """
        height = 220 if is_double else 140

        # print(plate_number, height, bg_color, is_double)
        number_xy = self.get_location_multi(plate_number, height)
        img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))
        img_plate_model = cv2.resize(img_plate_model, (440 if len(plate_number) == 7 else 480, height))

        for i in range(len(plate_number)):
            if len(plate_number) == 8:
                font_img = self.font_imgs['green_{}'.format(plate_number[i])]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                else:
                    if i < 2:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]

            if plate_number[i] in ['警',]:
                is_red = True
            elif (i == 1 or i ==0) and bg_color == 'white_army':
                # army plate
                is_red = True
            else:
                is_red = False

            if enhance:
                k = np.random.randint(1, 5)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

            img_plate_model = copy_to_image_multi(img_plate_model, font_img,
                                                  number_xy[i, :], bg_color, is_red)

        # is_double = 'double' if is_double else 'single'
        kernel = random.randint(3,9)
        img_plate_model = cv2.blur(img_plate_model, (kernel, kernel))

        if random.random()<0.7:
            img_plate_model = sp_noise(img_plate_model, prob=0.05)
        # elif random.random()<0.3:
        #     img_plate_model = gasuss_noise(img_plate_model, mean=0, var=0.01)

        return img_plate_model


if __name__ == '__main__':
    
    generator = MultiPlateGenerator('./license_plate_generator/plate_model', './license_plate_generator/font_model')
    lp =  generator.generate_plate_special('闽AAAAAA', 'blue', is_double=False)
    cv2.imwrite('1.jpg', lp)