import torch
import numpy as np
from numba import jit

# Refer: https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
class Guided_Attention_Loss(torch.nn.Module):
    def __init__(self, sigma= 0.2):
        super(Guided_Attention_Loss, self).__init__()
        self.sigma = sigma

    def forward(self, attentions, query_lengths, key_lengths):
        return (attentions * self.Get_Soft_Masks(query_lengths, key_lengths).to(attentions.device)).mean()

    def Get_Soft_Masks(self, query_lengths, key_lengths):
        query_lengths = query_lengths.cpu().numpy()
        key_lengths = key_lengths.cpu().numpy()
        max_Query_Length = max(query_lengths)
        max_Key_Length = max(key_lengths)
        masks = np.stack([
            _Calc_Soft_Masks(query_Length, max_Query_Length, key_Length, max_Key_Length, self.sigma).T
            for query_Length, key_Length in zip(query_lengths, key_lengths)
            ], axis= 0)

        return torch.FloatTensor(masks)

@jit(nopython=True)
def _Calc_Soft_Masks(query_length, max_query_length, key_length, max_key_length, sigma= 0.2):
    mask = np.zeros((max_query_length, max_key_length), dtype=np.float32)
    for query_Index in range(query_length):
        for key_Index in range(key_length):
            mask[query_Index, key_Index] = 1 - np.exp(-(query_Index / query_length - key_Index / key_length) ** 2 / (2 * sigma ** 2))

    return mask