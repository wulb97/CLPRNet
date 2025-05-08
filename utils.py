import numpy as np
import random
import cv2

from license_plate_generator.generate_multi_plate import MultiPlateGenerator
from license_plate_generator.plate_number import random_select, generate_plate_number_white, generate_plate_number_yellow_xue
from license_plate_generator.plate_number import generate_plate_number_black_gangao, generate_plate_number_black_shi, generate_plate_number_black_ling
from license_plate_generator.plate_number import generate_plate_number_blue, generate_plate_number_yellow_gua
from license_plate_generator.plate_number import letters, digits

plate_generator = MultiPlateGenerator('./license_plate_generator/plate_model', './license_plate_generator/font_model')

provinces1 = ["桂",]
provinces2 = ["京","冀","吉","晋","津","浙","湘","琼","甘","皖","粤","藏","豫","贵","闽","陕","青","鲁","辽","新","蒙","黑","云","宁","沪","渝","苏","赣","鄂"]

def resize(img, lurd_list, vertices_label_list):

    lurd_list_numpy = np.array(lurd_list).reshape((-1,2))
    leftUp = [np.min(lurd_list_numpy[:,0]), np.min(lurd_list_numpy[:,1])]
    rightDown = [np.max(lurd_list_numpy[:,0]), np.max(lurd_list_numpy[:,1])]

    height, width, _ = img.shape
    if (rightDown[1] - leftUp[1])/height < 60/1080 or len(lurd_list)>1:
        if len(lurd_list)>1:
            min_size = (rightDown[1] - leftUp[1]) if (rightDown[1] - leftUp[1])>(rightDown[0] - leftUp[0]) else (rightDown[0] - leftUp[0])
        else:
            min_size = (rightDown[1] - leftUp[1])*3 if (rightDown[1] - leftUp[1])>(rightDown[0] - leftUp[0]) else (rightDown[0] - leftUp[0])*3
        cx = (leftUp[0] + rightDown[0]) // 2
        cy = (leftUp[1] + rightDown[1]) // 2
        x1 = cx - min_size//2 if cx-min_size//2>0 else 0
        x2 = cx + min_size//2 if cx+min_size//2<width else width
        y1 = cy - min_size//2 if cy-min_size//2>0 else 0
        y2 = cy + min_size//2 if cy+min_size//2<height else height
        x1 = random.randint(0, x1)
        x2 = random.randint(x2, width)
        y1 = random.randint(0, y1)
        y2 = random.randint(y2, height)  
        img = img[y1:y2, x1:x2, :]

        for vertices_label in vertices_label_list:
            for i in vertices_label:
                i[0] = i[0] - x1
                i[1] = i[1] - y1

        for lurd in lurd_list:
            for i in lurd:
                i[0] = i[0] - x1
                i[1] = i[1] - y1

    elif (rightDown[1] - leftUp[1])/height > 100/1080 and not len(lurd_list)>1:
        margin_x = int(width*random.random()/2)
        margin_y = int(height*random.random()/2)
        x1 = random.randint(0,margin_x)
        y1 = random.randint(0,margin_y)
        img2 = np.zeros((height+margin_y*2, width+margin_x*2, 3))
        img2[y1:y1+height,x1:x1+width,:] = img
        img = img2

        for vertices_label in vertices_label_list:
            for i in vertices_label:
                i[0] = i[0] + x1
                i[1] = i[1] + y1

        for lurd in lurd_list:
            for i in lurd:
                i[0] = i[0] + x1
                i[1] = i[1] + y1

    return img, lurd_list, vertices_label_list

def warp(img, lurd_list, vertices_label_list):
        
        rows, cols, _ = img.shape
        if random.random()<0.5:
            M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),random.randint(-30,30),1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            new_vertices_label_list = []
            for vertices in vertices_label_list:
                new_vertices = []
                for i in vertices:
                    px = M[0,0]*i[0]+M[0,1]*i[1]+M[0,2]
                    py = M[1,0]*i[0]+M[1,1]*i[1]+M[1,2]
                    new_vertices.append([int(px), int(py)])
                new_vertices_label_list.append(new_vertices)
        else:
            pts1 = [[0,0],[0,rows],[cols,rows],[cols,0]]
            if random.random()<0.4:
                pts2 = []
                for i in pts1:
                    pts2.append([i[0]+random.randint(-cols//6,cols//6), i[1]+random.randint(-rows//6,rows//6)]) 
            else:
                dx = random.randint(0,cols//3)
                dy = random.randint(-rows//3,rows//3)
                pts2 = [[0+dx,0+dy],[0+dx,rows+dy],[cols,rows],[cols,0]]
            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(img,M,(cols,rows))
            new_vertices_label_list = []
            for vertices in vertices_label_list:
                new_vertices = []
                for i in vertices:
                    px = (M[0][0]*i[0] + M[0][1]*i[1] + M[0][2]) / ((M[2][0]*i[0] + M[2][1]*i[1] + M[2][2]))
                    py = (M[1][0]*i[0] + M[1][1]*i[1] + M[1][2]) / ((M[2][0]*i[0] + M[2][1]*i[1] + M[2][2]))
                    new_vertices.append([int(px), int(py)])
                new_vertices_label_list.append(new_vertices)
        
        tmp = np.array(new_vertices_label_list).reshape(-1,2)
        minx = np.min(tmp[:,0])
        maxx = np.max(tmp[:,0])
        miny = np.min(tmp[:,1])
        maxy = np.max(tmp[:,1])
        if minx<0 or maxx>cols or miny<0 or maxy>rows:
            return img, lurd_list, vertices_label_list
        else:
            new_lurd_list = []
            for vertices in new_vertices_label_list:
                tmp = np.array(vertices)
                minx = np.min(tmp[:,0])
                maxx = np.max(tmp[:,0])
                miny = np.min(tmp[:,1])
                maxy = np.max(tmp[:,1])
                new_lurd_list.append([[minx, miny], [maxx, maxy]])
            return dst, new_lurd_list, new_vertices_label_list
        
def generate_plate_number(color='norm'):
    '''
    color: 'green' 'double' 'lack' 'norm' 'multi'
    '''
        
    if color=='green':
        # 新能源
        plate_number = generate_plate_number_blue(8)
    elif color=='double':
        # 双层黄牌车、黄牌挂车
        if random.random()<0.7:
            plate_number = generate_plate_number_blue()
        else:
            if random.random()<0.5:
                plate_number = generate_plate_number_yellow_gua()
            else:
                plate_number = generate_plate_number_white()
    elif color=='lack':
        plate_number = generate_plate_number_blue(provinces=provinces1)
    elif color=='multi': 
        if random.random()<0.8:
            if random.random()<0.9:
                # 蓝牌, 黄牌车
                plate_number = generate_plate_number_blue(provinces=provinces1*10 + provinces2 + ['川'])
            else:
                # 绿
                plate_number = generate_plate_number_blue(8)
        else:
            if random.random()<0.7:
                generate_plate_number_funcs = [generate_plate_number_white,
                                            generate_plate_number_yellow_xue,]
            else:
                generate_plate_number_funcs = [
                                        generate_plate_number_black_gangao,
                                        generate_plate_number_black_shi,
                                        generate_plate_number_black_ling]
            plate_number = random_select(generate_plate_number_funcs)()
    else:
        if random.random()<0.8:
            if random.random()<0.9:
                # 蓝牌, 黄牌车
                plate_number = generate_plate_number_blue()
            else:
                # 绿
                plate_number = generate_plate_number_blue(8)
        else:
            if random.random()<0.7:
                generate_plate_number_funcs = [generate_plate_number_white,
                                            generate_plate_number_yellow_xue,]
            else:
                generate_plate_number_funcs = [
                                        generate_plate_number_black_gangao,
                                        generate_plate_number_black_shi,
                                        generate_plate_number_black_ling]
            plate_number = random_select(generate_plate_number_funcs)()

    # 车牌底板颜色
    if color=='double':
        is_double = True
        bg_color = 'yellow'
    else:
        bg_color = random_select(['blue']*5 + ['yellow'])
        is_double = False

    if len(plate_number) == 8:
        bg_color = random_select(['green_car'] * 5 + ['green_truck'])
    elif len(set(plate_number) & set(['港', '澳'])) > 0:
        bg_color = 'black'
    elif '警' in plate_number:
        bg_color = 'white'
    elif len(set(plate_number) & set(['学', '挂'])) > 0:
        bg_color = 'yellow'
    elif '使' in plate_number:
        bg_color = 'black_shi'
    elif '领' in plate_number:
        bg_color = 'black_ling'
    elif plate_number[0] in letters:
        bg_color = 'white_army'

    return plate_number, bg_color, is_double

def replace(img, vertices, leftUp, rightDown, data_type='norm', resize=0.2):
    '''
    data_type: 'green' 'double' 'lack' 'norm' 'multi'
    '''
    plate_number, bg_color, is_double = generate_plate_number(data_type)
    lp = plate_generator.generate_plate_special(plate_number, bg_color, is_double)

    old_lp = img[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0],:]
    mu = np.mean(old_lp)
    sigma = np.std(old_lp)
    mu2 = np.mean(lp)
    sigma2 = np.std(lp)
    lp = (lp-mu2)/sigma2
    lp = lp*sigma + mu
    lp = np.clip(lp, 1, 255)
    lp = lp.astype('uint8')

    if random.random()<resize:
        rate = random.randint(2,6)
        lp2 = cv2.resize(lp, (lp.shape[1]//rate, lp.shape[0]//rate), interpolation=cv2.INTER_NEAREST)
        lp = cv2.resize(lp2, (lp.shape[1], lp.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    rows, cols, _ = lp.shape
    pts1 = [[cols,rows],[0,rows],[0,0],[cols,0]]
    pts2 = vertices
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    rows, cols, _ = img.shape
    dst = cv2.warpPerspective(lp,M,(cols,rows))
    mask = cv2.inRange(dst, np.array([0,0,1]), np.array([255,255,255]))
    img[mask != 0] = [0, 0, 0]
    img = cv2.add(img, dst)
    tmp = np.array(vertices)
    minx = np.min(tmp[:,0])
    maxx = np.max(tmp[:,0])
    miny = np.min(tmp[:,1])
    maxy = np.max(tmp[:,1])

    if data_type=='double':
        bg_color = 'double'
    return img, vertices, [minx, miny], [maxx, maxy], plate_number, bg_color

def gauss(kernel_size:tuple, mu=(0,0), sigma=(1,1), rou=0, scale = 3):
    x, y = np.meshgrid(np.linspace(-scale, scale, kernel_size[1]), np.linspace(-scale, scale, kernel_size[0]))
    gauss = 1/(2*np.pi*sigma[0]*sigma[1]*np.sqrt(1-rou**2)) * np.exp(-1/(2*(1-rou**2)) * ( (x-mu[0])**2/sigma[0]**2 -2*rou*(x-mu[0])*(y-mu[1])/(sigma[0]*sigma[1]) +  (y-mu[1])**2/sigma[1]**2) )
    return gauss

def add_random_gauss_brightness(im):
    o = gauss(im.shape[:2],mu=(random.randint(-30,30)/10,random.randint(-30,30)/10), sigma=(random.random()*2,random.random()*2))
    g = o/(np.max(o)+0.01)*random.randint(0,200)
    g = np.expand_dims(g,2).repeat(3,2).astype('uint8')
    im = cv2.add(im,g)
    return im

def blur(im):
    flag = True
    while flag:
        degree = random.randint(1,3)
        angle = random.randint(0,360)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1) 
        motion_blur_kernel = np.diag(np.ones(degree)) 
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree)) 
        motion_blur_kernel = motion_blur_kernel / degree 
        if np.max(motion_blur_kernel)>0:
            flag = False
    blurred = cv2.filter2D(im, -1, motion_blur_kernel) # convert to uint8 
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX) 
    blurred = np.array(blurred, dtype=np.uint8) 
    return blurred

def add_random_erase(im):
    scale = 2
    for i in range(random.randint(0,10)):
        y = random.randint(0, im.shape[0])
        im[y:y+scale,:,:] = 0
    for i in range(random.randint(0,10)):
        x = random.randint(0, im.shape[1])
        im[:,x:x+scale,:] = 0
    return im

def iou(box, box2):
    box_area = (box[2]-box[0])*(box[3]-box[1])
    other_boxes_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    x1 = max(box[0],box2[0])
    y1 = max(box[1],box2[1])
    x2 = min(box[2],box2[2])
    y2 = min(box[3],box2[3])
    Min = 0
    w,h = max(Min,x2-x1),max(Min,y2-y1)
    overlap_area = w*h
    iou = overlap_area / (box_area+other_boxes_area-overlap_area)
    return iou


def crop(img, lurd_list, vertices_label_list):

    lurd_list_numpy = np.array(lurd_list).reshape((-1,2))
    leftUp = [np.min(lurd_list_numpy[:,0]), np.min(lurd_list_numpy[:,1])]
    rightDown = [np.max(lurd_list_numpy[:,0]), np.max(lurd_list_numpy[:,1])]

    height, width, _ = img.shape
    if len(lurd_list_numpy) == 2:
        min_size = (rightDown[1] - leftUp[1])*5 if (rightDown[1] - leftUp[1])>(rightDown[0] - leftUp[0]) else (rightDown[0] - leftUp[0])*5
    else:
        min_size = (rightDown[1] - leftUp[1])*1.1 if (rightDown[1] - leftUp[1])>(rightDown[0] - leftUp[0]) else (rightDown[0] - leftUp[0])*1.1
    min_size = int(min_size)
    cx = (leftUp[0] + rightDown[0]) // 2
    cy = (leftUp[1] + rightDown[1]) // 2
    x1 = cx - min_size//2 if cx-min_size//2>0 else 0
    x2 = cx + min_size//2 if cx+min_size//2<width else width
    y1 = cy - min_size//2 if cy-min_size//2>0 else 0
    y2 = cy + min_size//2 if cy+min_size//2<height else height 
    img = img[y1:y2, x1:x2, :]

    for vertices_label in vertices_label_list:
        for i in vertices_label:
            i[0] = i[0] - x1
            i[1] = i[1] - y1

    for lurd in lurd_list:
        for i in lurd:
            i[0] = i[0] - x1
            i[1] = i[1] - y1

    return img, lurd_list, vertices_label_list

def clean(img, lurd_list):

    new_img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    new_img = cv2.resize(new_img, (img.shape[1], img.shape[0]))
    height, width, _ = img.shape
    for lurd in lurd_list:
        leftUp, rightDown = lurd
        min_high = (rightDown[1] - leftUp[1])*20 
        min_width = (rightDown[0] - leftUp[0])*7 
        cx = (leftUp[0] + rightDown[0]) // 2
        cy = (leftUp[1] + rightDown[1]) // 2
        x1 = cx - min_width//2 if cx-min_width//2>0 else 0
        x2 = cx + min_width//2 if cx+min_width//2<width else width
        y1 = cy - min_high//2 if cy-min_high//2>0 else 0
        y2 = cy + min_high//2 if cy+min_high//2<height else height 
        new_img[y1:y2, x1:x2, :] = img[y1:y2, x1:x2, :]
    return new_img

if __name__ == '__main__':
    l = [[[500,500],[550,550]]]
    l2 = np.array(l)
    l[0][0][0] = 0
    print(l)
    print(l2)
