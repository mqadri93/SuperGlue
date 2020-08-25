import sys

sys.path.append('./')

import numpy as np
import torch
import os, sys
import cv2
import math, time
import datetime

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

import models.utils as uu
# from skimage import io, transform
# from skimage.color import rgb2gray
#
# from models.superpoint import SuperPoint
# from models.utils import frame2tensor, array2tensor


class SIFTDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, image_path, image_list = None, nfeatures = 1024):

        print('Using SIFT dataset')

        self.image_path = image_path

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(image_path)
                if name.endswith('jpg') or name.endswith('png') ]

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        # self.sift = cv.xfeatures2d.SIFT_create()
        # self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)

        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        # load precalculated correspondences
        # data = np.load(self.files[idx], allow_pickle=True)

        # Read image
        #print(os.path.join(self.image_path, self.image_names[idx]))
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)

        # 使用 IO 读取图像 rgb
        # rgb_img = io.imread(file_name)
        # image = rgb2gray(rgb_img)
        sift = self.sift
        width, height = image.shape[:2]
        # max_size = max(width, height)
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))  # return an image type

        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        #kp1 = kp1[:kp1_num]
        #kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1]).astype(
            np.float32)  # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2]).astype(np.float32)

        if len(kp1) < 1 or len(kp2) < 1:
            # print("no kp: ",file_name)
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.float32),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.float32),
                'descriptors0': torch.zeros([0, 2], dtype=torch.float32),
                'descriptors1': torch.zeros([0, 2], dtype=torch.float32),
                'image0': image,
                'image1': warped,
                'file_name': self.image_names[idx]
            }
            #     descs1 = np.zeros((1, sift.descriptorSize()), np.float32)
        # if len(kp2) < 1:
        #     descs2 = np.zeros((1, sift.descriptorSize()), np.float32)

        scores1_np = np.array([kp.response for kp in kp1], dtype=np.float32)  # confidence of each key point
        scores2_np = np.array([kp.response for kp in kp2], dtype=np.float32)

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]
        scores1_np = scores1_np[0:kp1_num]
        scores2_np = scores2_np[0:kp2_num]

        matched = self.matcher.match(descs1, descs2)
        
        # Match descriptors.
        '''
        kp1, des1 = sift.detectAndCompute(image,None)
        kp2, des2 = sift.detectAndCompute(warped,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matched = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            #if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(image,kp1,warped,kp2,good,None, flags=2)
        img4 = cv2.drawMatches(image, kp1, warped, kp2, matched, None, flags=2)
        #img3 = cv2.drawMatches(image, kp1_np, warped, kp2_np, good, flags=2)
        cv2.imwrite("matches.jpg", img3)
        cv2.imwrite("img4.jpg", img4)
        cv2.imwrite("orig.jpg", image)
        cv2.imwrite("warp.jpg", warped)
        #sys.exit()
        '''

        num_kp1 = len(kp1_np)
        num_kp2 = len(kp2_np)
        all_matches = np.zeros((num_kp1+1, num_kp2+1))

        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
        num =0
        used_train = []
        used_query = []
        plot_matches = []
        for m in matched:
            trainIdx = m.trainIdx
            queryIdx = m.queryIdx
            projected = kp1_projected[queryIdx]
            detected = kp2_np[trainIdx]
            if np.linalg.norm(projected - detected) < 5:
                if all_matches[queryIdx, trainIdx] == 1:
                    continue
                if trainIdx in used_train or queryIdx in used_query: 
                    continue
                used_train.append(trainIdx)
                used_query.append(queryIdx)
                plot_matches.append((kp1_np[queryIdx], kp2_np[trainIdx]))
                num+=1
                all_matches[queryIdx, trainIdx] = 1
        
        for ll in range(all_matches.shape[0]):
            if all(all_matches[ll,:] == 0):
                all_matches[ll, -1] = 1

        for ll in range(all_matches.shape[1]):
            if all(all_matches[:,ll] == 0):
                all_matches[-1, ll] = 1

        used_train = list(set(used_train))
        used_query = list(set(used_query))
        assert len(used_query) == num 
        assert len(used_train) == num

        assert np.count_nonzero(all_matches[:-1,:-1]) == num
        #uu.drawkp(image, warped, plot_matches, 1)
        #time.sleep(5)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        # 归一化+通道数扩充一维
        image = torch.from_numpy(image / 255.).float().unsqueeze(0).cuda()
        warped = torch.from_numpy(warped / 255.).float().unsqueeze(0).cuda()

        #sys.exit()
        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': self.image_names[idx],
        }

