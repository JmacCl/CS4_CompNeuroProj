import torch
import numpy as np

def convert_mask(mask, model:bool, threshold=0.6):
    if model:
        mask = torch.softmax(mask, dim=0)
    argmax_indices = torch.argmax(mask, dim=0)

    return argmax_indices


def convert_batch_train(data, mri):
    squeeze = data.squeeze(0)
    take = squeeze[mri]
    np_ver = take.numpy()
    return np_ver

def convert_num_to_string(num: int) -> str:
    string_num = str(num)
    if len(string_num) == 1:
        return "00" + string_num
    elif len(string_num) == 2:
        return "0" + string_num
    else:
        return string_num



# def remove_useless(input, labels):
#     """
#     This function will output the data files where there is no useless information
#     :param input:
#     :param labels:
#     :return:
#     """
#     new_input = []
#     new_target = []
#     for idx, label in enumerate(labels):
#         conv_label = convert_mask(label, False)
#         if not torch.all(conv_label == 0):
#             new_input.append(input[idx])
#             new_target.append(labels[idx])
#     return new_input, new_target


# def map_and_plot(x, model_output, batch):
#     # Ensure the tensor is in the range [0, 1] for proper visualization
#     x_output = torch.clamp(x, 0, 1)
#     model_output = torch.clamp(model_output, 0, 1)
#
#     if batch == 5:
#         # Select batch index and channel for visualization
#         batch_index = 0
#         channel_index = 0
#
#         # Extract the image from the tensor
#         image_x = x_output[batch_index, channel_index].numpy()
#         print(image_x.shape)
#         image_array = model_output[batch_index, channel_index].numpy()
#
#         # Display the image using matplotlib
#         plt.imshow(image_x, cmap='gray')  # Choose a colormap if needed
#         plt.axis('off')  # Turn off axis labels
#         plt.show()
#
#         # Display the image using matplotlib
#         plt.imshow(image_array, cmap='viridis')  # Choose a colormap if needed
#         plt.axis('off')  # Turn off axis labels
#         plt.show()

