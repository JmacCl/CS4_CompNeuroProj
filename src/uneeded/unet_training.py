import os

import nibabel as nib
import numpy as np
import tensorflow as tf

from keras_unet_collection import models
from src.experiments.utility_functions import convert_num_to_string
from src.uneeded.utils.brain_data_preprocessing import brain_data_preprocessing

TRAIN_PATH = "/src/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

N_epoch = 100  # number of epoches
N_batch = 155  # number of batches per epoch
tol = 0 # current early stopping patience
max_tol = 3 # the max-allowed early stopping patience
min_del = 0 # the lowest acceptable loss value reduction



# Define the inputs tensor shape
input_shape = (240, 240, 1)
input_tensor = tf.keras.layers.Input(shape=input_shape)

# Create the UNet model
TestUnet = models.vnet_2d((240, 240, 1),
                          filter_num=[16, 32, 64, 128], n_labels=1,
                          res_num_ini=1, res_num_max=3,
                          activation='PReLU',
                          output_activation='Sigmoid',
                          batch_norm=True, pool=False,
                          unpool=False, name='vnet')

# TestUnet = models.unet_2d((240, 240, 1),
#                           [16, 32, 64, 128], n_labels=1,
#                       stack_num_down=1, stack_num_up=1,
#                       activation='ReLU', output_activation='Sigmoid',
#                       batch_norm=False, pool=False, unpool=False, name='unet')

TestUnet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

loss_results = []

# loop over epoches and select a given t2 fluid data set
for epoch in range(N_epoch):

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

    train_string = "BraTS20_Training_" + convert_num_to_string(1 + epoch)

    # Identify stings
    # t1_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t1.nii")
    # t1ce_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t1ce.nii")
    # t2_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t2.nii")
    flair_string = os.path.join(TRAIN_PATH, train_string, train_string + "_flair.nii")

    # Get data
    # data_input_t1: np.ndarray = nib.load(t1_string).get_fdata()
    # data_input_t1ce: np.ndarray = nib.load(t1ce_string).get_fdata()
    # data_input_t2: np.ndarray = nib.load(t2_string).get_fdata()
    data_input_flair: np.ndarray = nib.load(flair_string).get_fdata()
    seg: np.ndarray = nib.load(os.path.join(TRAIN_PATH, train_string, train_string + "_seg.nii")).get_fdata()
    print("Intersting" ,data_input_flair.shape)
    print(seg.shape)
    # loop over batches
    print(train_string)
    for step in range(N_batch):
        # Get data
        # processed_t1 = brain_data_preprocessing(data_input_t1[:, :, step])
        # processed_t1ce = brain_data_preprocessing(data_input_t1ce[:, :, step])
        # processed_t2 = brain_data_preprocessing(data_input_t2[:, :, step])
        processed_flair = brain_data_preprocessing(data_input_flair[:, :, step])

        if not (processed_flair is None):
            # train on batch
            main_data = processed_flair
            seg_data = seg[:, :, step]
            processed_data_input = tf.expand_dims(main_data, axis=-1)
            p_seg_input = tf.expand_dims(seg_data, axis=-1)
            print(processed_data_input.shape)
            print(p_seg_input.shape)
            loss_ = TestUnet.train_on_batch(processed_data_input, p_seg_input)
            loss_results.append(loss_)
            # print("Current loss", loss_)
        #         if np.isnan(loss_):
        #             print("Training blow-up")



print("finished")

# save validation and training_utils results,
# if validation diverges, in comparison to training_utils, may more data
# show simple model works, show that it can make predictions,
# qualatative results, loss druing training_utils, validation and training_utils loss
# as soon as I have results
# status report

SAVE_PATH = os.path.join("/", "src", "../experiments/saved_model")

# Save Model
TestUnet.save(os.path.join(SAVE_PATH,"2_label_flair_test", 'TestUnet.keras'))

