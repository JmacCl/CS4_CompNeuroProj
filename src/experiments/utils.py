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