import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses.focal import FocalLoss
from kornia.losses.dice import DiceLoss



# class DiceLoss(nn.Module):
#     def __init__(self, n_classes, softmax=True):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#         self.softmax = softmax
#
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def __loss_function(self, score, target):
#         metric_output = self.function(score, target)
#         return 1 - metric_output
#
#     def forward(self, inputs, target, weight=None):
#         # If a softmax layer is required for loggits
#         if self.softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         # target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes



class FocalDiceComboLoss(nn.Module):

    def __init__(self, dice_coeff, focal_coeff, f_alpha, f_gamma):
        """
        Init the loss
        """
        super(FocalDiceComboLoss, self).__init__()
        self.__check_coefficients(dice_coeff, focal_coeff)
        self.dice = DiceLoss()
        self.dice_coeff = dice_coeff
        self.focal = FocalLoss(f_alpha, f_gamma, reduction="mean")
        self.focal_coeff = focal_coeff

    def forward(self, y_pred, t_true):

        dice_result = self.dice_coeff * self.dice(y_pred, t_true)
        focal_result = self.focal_coeff * self.focal(y_pred, t_true)

        return dice_result + focal_result


    def __check_coefficients(self, coefficient_one, coefficient_two):

        if coefficient_one + coefficient_two != 1:
            raise ValueError("coefficients are incorrect, they must add up to 1")
