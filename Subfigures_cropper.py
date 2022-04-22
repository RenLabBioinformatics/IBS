from typing import Any, Union

import cv2
import numpy as np
from skimage import data,filters,segmentation,measure,morphology,color,io
from skimage import img_as_ubyte
import logging
import os
import shutil
from tensorflow.keras.preprocessing import  image
from PIL import Image
import  random
class FigCropper(object):
    """docstring for FigCropper"""
    def __init__(self, img=None):
        super(FigCropper, self).__init__()
        self.img = img

        """the final goal of this program is make region extent to be 1.
        which means, every region has beed filled to rectangle"""
        self.region_extent = 0

        # for a region,if img dim/ region < this, region will be swipe from crop mask
        self.min_region_area_ratio = 30
        self.sub_imgs = []
        self.crop_mask = None
        # 2021.6.5 redfish add
        self.drop_imgs = []
        self.saved_imgs = []


    def preprocess(self):
        if self.img is None:
            logging.error('can not crop img from None')
        self.img_gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.img_binary = 255-self.img_gray # reverse to make background black
        row, col = self.img.shape[:2]
        bottom = self.img[row-1:row,:]
        mean = cv2.mean(bottom)[0]
        ret, self.img_binary = cv2.threshold(self.img_gray, mean-20, 255, cv2.THRESH_BINARY_INV)


        # dialated
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(self.img_binary, kernel,iterations=1)
        dilated_area = np.shape(dilated)[0]*np.shape(dilated)[1]
        self.crop_mask = dilated

    ########################################
    # Inner operations
    ########################################
    def swipe_region(self,region):
        minr, minc, maxr, maxc = region.bbox
        self.crop_mask[minr:maxr,minc:maxc] = 0

    def fill_region(self,region):
        minr, minc, maxr, maxc = region.bbox
        self.crop_mask[minr:maxr,minc:maxc] = 255

    def crop_region(self,region):
        minr, minc, maxr, maxc = region.bbox
        fig = self.img[minr:maxr,minc:maxc]
        return fig,(minr, minc, maxr, maxc)

    def swipe_small_regions(self):
        temp = min(self.img.shape[:2])
        min_area = (temp/self.min_region_area_ratio)**2
        for region in measure.regionprops(self.markers):
            if region.area < min_area:
                self.swipe_region(region)

    def fill_irregular_regions(self):
        self.region_extent = 1 # init extent with 1
        self.markers = measure.label(self.crop_mask)
        for region in measure.regionprops(self.markers):
            self.fill_region(region)
            self.region_extent = min(self.region_extent,region.extent)

    ########################################
    # Interface operations
    ########################################
    def get_crop_mask(self):
        self.crop_mask = None
        self.region_extent = 0
        self.preprocess()
        self.markers = measure.label(self.crop_mask)
        self.swipe_small_regions()
        for i in range(100):
            if self.region_extent < 1:
                self.fill_irregular_regions()
        return self.crop_mask

    def get_colored_mask(self):
        self.get_crop_mask()
        ret, markers = cv2.connectedComponents(self.crop_mask)
        image_label_overlay =color.label2rgb(markers, image=self.crop_mask)
        image_label_overlay = img_as_ubyte(image_label_overlay)
        return image_label_overlay


    def get_cropped_imgs(self):
        self.sub_imgs = []
        self.get_crop_mask()
        self.markers = measure.label(self.crop_mask)
        for region in measure.regionprops(self.markers):
            fig,coords = self.crop_region(region)
            self.sub_imgs.append([fig,coords])
        return self.sub_imgs
    # test
    def get_cropped_imgs2(self):
        self.sub_imgs = []
        self.get_crop_mask()
        self.markers = measure.label(self.crop_mask)
        # cv2.imshow('test',self.img)
        # cv2.waitKey(0)
        # minr, minc, maxr, maxc =  coords
        # sub_img=[]
        for region in measure.regionprops(self.markers):
            fig,coords = self.crop_region(region)
            self.sub_imgs.append([fig,coords])
            # 1:5 5:1 aspect ratio
            # area
            # sub_img.append([fig,coords])
        if len(self.sub_imgs)==0:
            return None
        else:
            self.swip_subimg_by_area_ratio()
            return self.saved_imgs


    def swip_subimg_by_area_ratio(self):
        area = []
        # subimgs,coords = self.sub_imgs[0],self.sub_imgs[1]
        ratio =[]
        for i in range(len(self.sub_imgs)):
            coords = self.sub_imgs[i][1]
            area.append((coords[2] - coords[0])*(coords[3] - coords[1]))
            col = coords[2] - coords[0]
            row = coords[3] - coords[1]
            if row > col:
                ratio.append(row/col)
            else:
                ratio.append(col/row)
        max_area = 0.0
        if len(area) != 1:
            max_area = max(area)
        else:
            max_area = area[0]

        for j in range(len(area)):
            temp_area = area[j]
            temp_ratio = ratio[j]

            if temp_area/max_area < 0.05 :
                self.drop_imgs.append([self.sub_imgs[j][0],self.sub_imgs[j][1]])
            elif temp_area/max_area > 0.05:
                if (temp_ratio >5.0 and temp_ratio <8.0):

                    coords = self.sub_imgs[j][1]
                    col = coords[2] - coords[0]
                    row = coords[3] - coords[1]
                    if col > 25 or row > 25:
                        self.saved_imgs.append([self.sub_imgs[j][0], self.sub_imgs[j][1]])
                    else:
                        self.drop_imgs.append([self.sub_imgs[j][0], self.sub_imgs[j][1]])
                elif temp_ratio > 8.0:
                    coords = self.sub_imgs[j][1]
                    col = coords[2] - coords[0]
                    row = coords[3] - coords[1]
                    if col < 10.0 or row < 10.0:
                        self.drop_imgs.append([self.sub_imgs[j][0], self.sub_imgs[j][1]])
                    else:
                        self.saved_imgs.append([self.sub_imgs[j][0],self.sub_imgs[j][1]])
                elif temp_ratio <= 5.0:
                    self.saved_imgs.append([self.sub_imgs[j][0], self.sub_imgs[j][1]])
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg') or f.endswith('.png'):
                fullname = os.path.join(root, f)
                yield fullname
def findAllFile_returnName(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg') or f.endswith('.png'):
                fullname = os.path.join(root, f)
                yield fullname,f
def getCropImages(root_name):
    root_path = "F:\\IBS_data\\Science\\" + root_name + "\\"
    dir_list = os.listdir(root_path)

    for i in range(len(dir_list)):
        path = root_path + dir_list[i]
        if dir_list[i] != 'record':
            count = 0
            count_drop = 0
            for temp_path in findAllFile(path):
                if temp_path.find('croppedData') == -1 and temp_path.find('failed') == -1 :

                    temp_name = temp_path.split("\\")[4]
                    img = cv2.imread(temp_path)
                    fig_cropper = FigCropper(img)
                    # cropped_data = fig_cropper.get_cropped_imgs_SlidingWindow()
                    cropped_datas = fig_cropper.get_cropped_imgs2()
                    dropped_data = fig_cropper.drop_imgs
                    save_path_test = 'F:\\IBS_data\\Science\\CroppedData\\'+root_name+'\\'
                    drop_path_test = 'F:\\IBS_data\\Science\\DroppedData\\' + root_name + '\\'
                    failed_path = 'F:\\IBS_data\\Science\\failed\\'

                    if cropped_datas != None:
                        for j in range(len(cropped_datas)):

                            cv2.imwrite(save_path_test + "\\" + temp_name + "_" + str(count) + '.jpg',
                                        cropped_datas[j][0])
                            count += 1
                        for k in range(len(dropped_data)):

                            cv2.imwrite(drop_path_test + "\\" + temp_name + "_" + str(count_drop) + '.jpg',
                                        dropped_data[k][0])
                            count_drop += 1
                    else:
                        cv2.imwrite(failed_path + "\\" + temp_name + "_full.jpg",
                                    img)


            print(dir_list[i] + " crop finish")
    print(root_name + " crop finish")
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = abs((sh-new_h)/2)
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = abs((sw-new_w)/2)
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img1 = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img2 = cv2.copyMakeBorder(scaled_img1, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    # cv2.imshow('test',scaled_img1)
    # cv2.imshow('test2', scaled_img2)
    # cv2.waitKey(0)

    return scaled_img2
def findBackGround(mask):
    result_x = []
    result_y = []
    row = len(mask)
    col = len(mask[0])
    for i in range(row):
        for j in range(col):
            if mask[i][j] == 0:
                result_x.append(i)
                result_y.append(j)
    return result_x,result_y
def preprocess4background(img):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        original_area = np.shape(img)[0]*np.shape(img)[1]
        # img_binary = 255-img_gray # reverse to make background black
        ret, img_binary = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)

        # dialated
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(img_binary, kernel,iterations=2)
        dilated_area = np.shape(dilated)[0]*np.shape(dilated)[1]
        if dilated_area/original_area > 0.5:
            crop_mask = img_binary
        else:
            crop_mask = dilated
        return crop_mask.tolist()
def resize_Pipline4Predict(dir):
    path = 'F:\\IBS_data\\Science\\CroppedData\\'+dir+'\\'
    save_path='F:\\IBS_data\\Science\\CroppedData\\resized\\'+dir+'\\'

    for temp_file,temp_filename in findAllFile_returnName(path):
            img = cv2.imread(temp_file)
            # get pad color
            mask = preprocess4background(img)
            index_x, index_y = findBackGround(mask)
            pad_color = 0
            if len(index_x) == 0 and len(index_y) == 0:
                pad_color = 255
            else:
                bottom = img[index_x, index_y]
                pad_color = cv2.mean(bottom)[0]

            scaled_sq_img = resizeAndPad(img, (256, 256), pad_color)
            cv2.imwrite(save_path + temp_filename, scaled_sq_img)
if __name__ == '__main__':
    file_name="ids"
    getCropImages(file_name)
    resize_Pipline4Predict(file_name)



