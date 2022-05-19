import cv2
import numpy as np
import math


def crop(src):
    """
    transverse_divide(src) -> panel
    .   @brief 手动裁剪出src中关键部位
    .   @param src 待裁剪的图像
    """
    crop_w, crop_h = src.shape[0:2]
    src = cv2.resize(src, (int(crop_h / 2), int(crop_w / 2)))
    roi = cv2.selectROI('Select your ROI', img=src, showCrosshair=True, fromCenter=False)
    crop_x, crop_y, crop_w, crop_h = roi
    # 显示ROI并保存图片
    if roi != (0, 0, 0, 0):
        # cv2.imshow('crop', src[y:y + h, x:x + w])
        # cv2.waitKey(0)
        return src[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    else:
        print('')
        return None


def my_sobel(src):
    # Sobel边缘检测
    dst_x = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=3)
    dst_y = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=3)
    dst_x = cv2.convertScaleAbs(dst_x)
    dst_y = cv2.convertScaleAbs(dst_y)
    return cv2.add(dst_x,dst_y)


def transverse_divide(src,points):
    """
    transverse_divide(src) -> [img1,img2,...]
    .   @brief 将src按Points横向分割，返回分割出的图片列表
    .   @param points自然数数组，元素个数必须是偶数
    .   @param src 待分割的图像，可以是单通道，也可以是多通道
    """
    if len(points) % 2 != 0:
        print('[Error]图像水平分割错误。必须是偶数个分割点')
        exit(-1)
    td_i = 0
    output = []
    while td_i < len(transverse):
        output.append(src[transverse[td_i]:transverse[td_i + 1]])
        td_i += 2
    return output


def portrait_divide(src):
    """
    transverse_divide(src) -> [img1,img2,...]
    .   @brief 将src纵向分割，返回分割出的图片列表
    .   @param src 待分割的图像，可以是单通道，也可以是多通道
    """
    pd_k = len(src[0])  # 图片的像素列数
    pd_h = len(src)  # 图片行数
    y = []  # 每列的白色像素个数，都在其中
    for i in range(0, pd_k):
        num_w = 0
        for j in range(0,pd_h):  # 统计这一列非零元素的个数
            if src[j,i] != 0:
                num_w += 1
        y.append(num_w)
    # plt.bar(x, y, facecolor='#000000', width=1)
    # plt.show()
    portrait = []
    before = 0
    for m in range(0, pd_k):
        if (before == 0 and y[m] != 0) or (before != 0 and y[m] == 0):
            portrait.append(m)
        before = y[m]
        m += 1
    n = 0
    if len(portrait) % 2 != 0:
        print('[Error]图像纵向分割错误。必须是偶数个分割点')
        exit(-1)
    output = []
    # print('portrait')
    # print(portrait)
    while n < len(portrait):
        output.append(src[:,portrait[n]:portrait[n + 1]])
        n += 2
    return output


def recognize(src):
    """
    recognize(src) -> num
    .   @brief 识别出图像src中包含的数字
    .   @param src 待识别的图像，只包含一个字符
    """
    # ker = np.ones((3,3), np.uint8)
    # cv2.imshow('dsb',src)
    # after_dilate = cv2.dilate(src, ker, 1)  # 数字太细了，需要膨胀一下
    height,width = src.shape  # 获取这个数字图片的宽、高
    # 首先根据宽高比识别出数字“1”
    if width/height < 0.33:  # 数字“1”的宽高比是非常小的
        return 1
    elif width/height > 0.65:  # 图像宽高比太大了，意味着这个可能不是数字，而是一个噪点
        return -2
    else:
        # 标记
        flag = [0,0,0,0,0,0,0]
        # 三条线穿过数字
        mid_x = int(width/2)
        top_y = int(height/3)
        bottom_y = top_y * 2
        # 划分区域
        area1 = (0,int(height/7))
        area2 = (int(height*3/7),int(height*4/7))
        area3 = (int(height*6/7),height)
        area4 = (0,int(width/5))
        area5 = (int(width*4/5),width)
        # 竖着
        for i in range(area1[0],area1[1]):
            if src[i][mid_x] == 255:
                flag[0] = 1
                break
        for i in range(area2[0],area2[1]):
            if src[i][mid_x] == 255:
                flag[1] = 1
                break
        for i in range(area3[0],area3[1]):
            if src[i][mid_x] == 255:
                flag[2] = 1
                break
        for i in range(area4[0],area4[1]):
            if src[top_y][i] == 255:
                flag[3] = 1
                break
        for i in range(area5[0],area5[1]):
            if src[top_y][i] == 255:
                flag[4] = 1
                break
        for i in range(area4[0],area4[1]):
            if src[bottom_y][i] == 255:
                flag[5] = 1
                break
        for i in range(area5[0],area5[1]):
            if src[bottom_y][i] == 255:
                flag[6] = 1
                break
        if flag == [1,1,1,1,1,0,1]:
            return 9
        elif flag == [1,1,1,1,1,1,1]:
            return 8
        elif flag == [1,0,0,1,1,0,1]:
            return 7
        elif flag == [1,1,1,1,0,1,1]:
            return 6
        elif flag == [1,1,1,1,0,0,1]:
            return 5
        elif flag == [0,1,0,1,1,0,1]:
            return 4
        elif flag == [1,1,1,0,1,0,1]:
            return 3
        elif flag == [1,1,1,0,1,1,0]:
            return 2
        elif flag == [1,0,1,1,1,1,1]:
            return 0
        return -1


def recognize2(src):
    height, width = src.shape  # 获取这个数字图片的宽、高
    # 首先根据宽高比识别出数字“1”
    if width / height < 0.33:  # 数字“1”的宽高比是非常小的
        return 1
    elif width / height > 0.65:  # 图像宽高比太大了，意味着这个可能不是数字，而是一个噪点
        return -2
    else:
        # 三条线穿过数字
        mid_x = int(width/2)
        top_y = int(height/3)
        bottom_y = top_y * 2
        # 划分区域
        area1 = (0,int(height/7))
        area2 = (int(height*3/7),int(height*4/7))
        area3 = (int(height*6/7),height)
        area4 = (0,int(width/5))
        area5 = (int(width*4/5),width)
        counter_1 = 0  # 一号线段交点计数器
        pos_1 = 0
        counter_2 = 0  # 二号线段交点计数器
        pos_2 = 0
        counter_3 = 0  # 三号线段交点计数器
        pos_3 = 0
        for i in range(area1[0],area1[1]):
            if src[i][mid_x] == 255:
                counter_1 += 1
                pos_1 += 4
                break
        for i in range(area2[0],area2[1]):
            if src[i][mid_x] == 255:
                counter_1 += 1
                pos_1 += 2
                break
        for i in range(area3[0],area3[1]):
            if src[i][mid_x] == 255:
                counter_1 += 1
                pos_1 += 1
                break
        if pos_1 == 4:
            return 7
        elif pos_1 == 2:
            return 4
        elif pos_1 == 5:
            return 0
        elif pos_1 == 7:  # 2,3,5,6,8,9
            for i in range(area4[0], area4[1]):
                if src[top_y][i] == 255:
                    counter_2 += 2
                    break
            for i in range(area5[0], area5[1]):
                if src[top_y][i] == 255:
                    counter_2 += 1
                    break
            for i in range(area4[0], area4[1]):
                if src[bottom_y][i] == 255:
                    counter_3 += 2
                    break
            for i in range(area5[0], area5[1]):
                if src[bottom_y][i] == 255:
                    counter_3 += 1
                    break
            if counter_2 == 1:  # 2,3
                if counter_3 == 1:
                    return 3
                elif counter_3 == 2:
                    return 2
                else:
                    print('[Error]三号线段探测出错！')
                    return -1
            elif counter_2 == 2:  # 5,6
                if counter_3 == 1:
                    return 5
                elif counter_3 == 3:
                    return 6
                else:
                    print('[Error]三号线段探测出错！')
                    return -1
            elif counter_2 == 3:  # 8,9
                if counter_3 == 1:
                    return 9
                elif counter_3 == 3:
                    return 8
                else:
                    print('[Error]三号线段探测出错！')
                    return -1
            else:
                print('[Error]二号线段探测出错！')
                return -1
        else:
            print('[Error]一号线段探测出错！')
            return -1


window = 'Window_name'
img = cv2.imread('../images/panel001.png')
img = crop(img)
if img is None:
    print('[Error]裁剪错误！')
    exit(-1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

'''图像预处理'''
img = cv2.medianBlur(img,5)  # 中值滤波


'''Canny边缘检测'''
edges = cv2.Canny(img,100,200,True)
kernel = np.ones((3,3),np.uint8)
dst = cv2.dilate(edges, kernel, 2)
# Canny之前必须二值化，此处是自适应阈值
dst = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

'''检测轮廓'''
contours,hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
# print('检测出'+str(len(contours))+'个轮廓')
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img,contours,0,(0,255,0),2)  # 画出轮廓
'''轮廓近似'''
epsilon = 0.1*cv2.arcLength(contours[0],True)  # 0.1倍的轮廓周长
approx = cv2.approxPolyDP(contours[0],epsilon,True)
# cv2.drawContours(img,approx,-1,(0,255,0),10)
# cv2.imshow('hello',img)
# cv2.waitKey(0)
'''图片矫正，透视变换'''
# 左上 左下 右上 右下
# pts1 = np.float32([[18,20],[181,11],[27,247],[200,238]])
if len(approx) != 4:
    print('[Error]没有找到面板区域！')
    exit(-1)
# 给点排序
point_array = []
p = [(approx[0][0][0],approx[0][0][1]),
     (approx[1][0][0],approx[1][0][1]),
     (approx[2][0][0],approx[2][0][1]),
     (approx[3][0][0],approx[3][0][1])]
# 找到左上
s = [p[0][0]+p[0][1],p[1][0]+p[1][1],p[2][0]+p[2][1],p[3][0]+p[3][1]]
tmp = s[0]
x = 0
for i in range(1,4):
    if tmp > s[i]:
        x = i  # 最小值下标
point_array.append(p[x])
p.remove(p[x])
# 找到左下
tmp = p[0][1]
x = 0
for i in range(1,3):
    if tmp > p[i][1]:
        x = i  # 最小值下标
point_array.append(p[x])
p.remove(p[x])
# 找到右上
tmp = p[0][0]
x = 0
for i in range(1,2):
    if tmp > p[i][0]:
        x = i  # 最小值下标
point_array.append(p[x])
p.remove(p[x])
# 找到右下
point_array.append(p[0])
pts1 = np.float32(point_array)
h1 = point_array[1][0] - point_array[0][0]
w1 = point_array[1][1] - point_array[0][1]
h2 = point_array[3][0] - point_array[1][0]
w2 = point_array[3][1] - point_array[1][1]
w = int(math.sqrt((h1**2)+(w1**2)))
h = int(math.sqrt((h2**2)+(w2**2)))
pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(img,M,(w,h))
dst = cv2.warpPerspective(img,M,(w,h))
# cv2.imshow('sdfsf',dst)
# cv2.waitKey(0)
'''去除显示区域的留白'''
# 上侧去除高度的45/210，下侧不用，右侧去除宽度的1/5，左侧去除宽度的1/16
left_margin = int(w / 16)
right_border = int(w * 0.8)
top_margin = int(h * 45 / 210)
dst_after_crop = dst[top_margin:h,left_margin:right_border]
# hell = cv2.adaptiveThreshold(cv2.cvtColor(dst_after_crop,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow('after crop',hell)  # 去除留白的
# cv2.waitKey(0)

'''对目标图片进行Canny'''
edges = cv2.Canny(dst_after_crop,45,70,True)
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(edges, kernel, 1)
# cv2.imshow('after canny and dilate',edges)
# cv2.waitKey(0)
'''字符分割'''
k = len(dilate)  # k为图片的像素行数
x = np.arange(k)
y = []  # 每行的白色像素个数，都在其中
for i in range(0, k):
    y.append(np.count_nonzero(dilate[i]))  # 统计这一行非零元素的个数
# plt.bar(x, y, facecolor='#000000', width=1)
# plt.show()
# 得到偶数个横坐标，两两一对，分割出图像
transverse = []
pre = 0
for i in range(0, k):
    if (pre == 0 and y[i] != 0) or (pre != 0 and y[i] == 0):
        transverse.append(i)
    pre = y[i]
    i += 1
num_img = transverse_divide(edges,transverse)  # 先水平分割，返回图片数组，元素个数就是数字的数量
number_amount = len(num_img)
j = 0
'''识别并输出'''
final_output = []
figures = []
while j < number_amount:
    # 分割出数字的每一位，即纵向分割
    # cv2.imshow('zongxiangfengeqian',num_img[j])
    # cv2.waitKey(0)
    single_character_array = portrait_divide(num_img[j])
    figures.append(single_character_array)
    position_value = []  # 当前行数字，识别出的由高位到低位每位的每一位
    total = 0
    time = 0
    for q in range(0,len(single_character_array)):
        e1 = cv2.getTickCount()
        x = recognize2(single_character_array[q])
        e2 = cv2.getTickCount()
        time += (e2 - e1) / cv2.getTickFrequency()
        if x == -1:
            print('----------没有数字-----------')
            # exit(0)
            total = -1
            break
        elif x == -2:  # 这个图像是的噪点
            print('-------------噪点---------------')
            total = -1
            break
        total *= 10
        total += x
    j += 1
    if total == -1:
        continue
    final_output.append(total)
    print('识别这个数字用了' + str(time*1000) + '毫秒')
print('final output is below:')
print(final_output)
