from skimage import io

import os
import h5py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


RAW_DATA = "C:\\Users\\james\\Uni_Projects\\CS4_CompNeuroProj\\data\\raw\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_026"

filenameT1Raw1 = os.path.join(RAW_DATA, "BraTS20_Training_026_t1.nii")
filenameT1CERaw1 = os.path.join(RAW_DATA, "BraTS20_Training_026_t1ce.nii")
filenameT2Raw1 = os.path.join(RAW_DATA, "BraTS20_Training_026_t2.nii")
filenameT2flair = os.path.join(RAW_DATA, "BraTS20_Training_026_flair.nii")
filenameSeg1 = os.path.join(RAW_DATA, "BraTS20_Training_026_seg.nii")

test_loadRaw1 = nib.load(filenameT1Raw1).get_fdata()
test_loadT1ce = nib.load(filenameT1CERaw1).get_fdata()
test_loadT2 = nib.load(filenameT2Raw1).get_fdata()
test_loadFlair = nib.load(filenameT2flair).get_fdata()
test_loadSeg1 = nib.load(filenameSeg1).get_fdata()



index = 48
ch0 = 40
ch1 = 40 + 128

cw0 = 30
cw1 = 30 + 128

#
test = test_loadSeg1[:, :, index]
print(test_loadSeg1.shape)
## Ignore no info

# print(new_shape.shape)
plt.imshow(test, )
plt.imshow(test, cmap="viridis")
#
# figure = plt.figure(test)
# ax = figure.add_subplot()
# ax.set_title("Tumor Segmentations")
# ax.legend(loc='best')
plt.show()
# # #
# testImg = test_loadRaw1[:, :, index]
# print(testImg.shape)
# # testReshape = testImg.reshape((128, 128, 128))
# plt.imshow(testImg, cmap="viridis" )
# plt.show()
#

# new_shape = np.copy(testImg)
# new_shape_2 = new_shape[cw0: cw1, ch0: ch1]
# plt.imshow(new_shape_2, cmap="gray" )
# plt.show()
# testImg = test_loadT1ce[:, :, index]
# plt.imshow(testImg, cmap="gray" )
# plt.show()
#
# testImg = test_loadT2[:, :, index]
# plt.imshow(testImg, cmap="gray" )
# plt.show()
#
# testImg = test_loadFlair[:, :, index]
# plt.imshow(testImg, cmap="gray" )
# plt.show()
# # Color bars, for segmented
#

TRAIN_PATH = os.path.join("/", "src/data\\raw\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData")

print(len(os.listdir(TRAIN_PATH)))
for fols in sorted(os.listdir(TRAIN_PATH)):
    # print(os.path.join(TRAIN_PATH, fols))
    fold_path = os.path.join(TRAIN_PATH, fols)
    if os.path.isdir(fold_path):
        for files in os.listdir(fold_path):
            print(files)
        print("\n")



