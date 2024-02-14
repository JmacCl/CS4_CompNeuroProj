import os
from keras_unet_collection import losses, utils
from keras_unet_collection import _model_unet_2d, _model_swin_unet_2d
from monai import transforms as tf
from PIL import Image

from data.raw.BraTS2020_TrainingData.MICCAI_BraTS2020_TrainingData import *
import nibabel as nib
import numpy as np
import tensorflow as tf

## Open CV for computer vision

PATH = "/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Create unet model

import tensorflow as tf
from keras_unet_collection import models

# Define the inputs tensor shape
input_shape = (240, 240, 1)
input_tensor = tf.keras.layers.Input(shape=input_shape)

# Create the UNet model
TestUnet = models.swin_unet_2d(
    input_size=input_shape,
    filter_num=[16, 32, 64, 128],
    n_labels=4,
    stack_num_down=2,
    stack_num_up=2,
    activation='ReLU',
    output_activation='Softmax',
    batch_norm=True,
    pool=True,
    unpool=True,
    name='unet'
)

TestUnet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# TestUnet.model.compile()

N_epoch = 369  # number of epoches
N_batch = 155  # number of batches per epoch
tol = 0 # current early stopping patience
max_tol = 3 # the max-allowed early stopping patience
min_del = 0 # the lowest acceptable loss value reduction


def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
    loss_iou = losses.iou_seg(y_true, y_pred)

    return loss_focal + loss_iou  # +loss_ssim

# loop over epoches and select a given t2 fluid data set
for epoch in range(N_epoch):
    print(epoch)

    # initial loss record
    # if epoch == 0:
    # data set and why its useful
    # segmentation vs classificaiton
    # classification to unet then to transformer to cs-unet
    # quan ying implementation questions
    #     temp_out = TestUnet.predict([valid_input])
    #     y_pred = temp_out[-1]
    #     record = np.mean(hybrid_loss(valid_target, y_pred))
    #     print('\tInitial loss = {}'.format(record))

    train_string = "BraTS20_Training_00" + str(epoch + 1)

    data_input: np.ndarray = nib.load(os.path.join(PATH, train_string, train_string + "_flair.nii")).get_fdata()
    seg: np.ndarray = nib.load(os.path.join(PATH, train_string, train_string + "_seg.nii")).get_fdata()
    # loop over batches
    print(train_string)
    for step in range(N_batch):
        # train on batch
        img1 = Image.fromarray(data_input[:, :, step], "L")
        input_img = utils.image_to_array(
            [img1],
            240,
            1)
        imgv = Image.fromarray(seg[:, :, step], "RGB")
        valid_img = utils.image_to_array(
            [imgv],
            240,
            4)
        loss_ = TestUnet.train_on_batch(input_img, valid_img)
        print("Current loss", loss_)
    #         if np.isnan(loss_):
    #             print("Training blow-up")

# Save Model
TestUnet.save('my_model.keras')


    # ** training loss is not stored ** #

    # # epoch-end validation
    # temp_out = TestUnet.predict([valid_input])
    # y_pred = temp_out[-1]
    # record_temp = np.mean(hybrid_loss(valid_target, y_pred))
    # # ** validation loss is not stored ** #
    #
    # # if loss is reduced
    # if record - record_temp > min_del:
    #     print('Validation performance is improved from {} to {}'.format(record, record_temp))
    #     record = record_temp;  # update the loss record
    #     tol = 0;  # refresh early stopping patience
    #     # ** model checkpoint is not stored ** #
    #
    # # if loss not reduced
    # else:
    #     print('Validation performance {} is NOT improved'.format(record_temp))
    #     tol += 1

