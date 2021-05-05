###################################################################################
# BSD 3-Clause License
#
# Copyright (c) 2020, MINDsLab Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################

from typing import Optional, Tuple
import torch

class ZoneoutLSTMCell(torch.nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, bias: bool= True, zoneout_rate: float= 0.1):
        super().__init__(input_size, hidden_size, bias)
        self.zoneout_rate = zoneout_rate
        
        self.dropout = torch.nn.Dropout(p= zoneout_rate)

        # initialize all forget gate bias of LSTM to 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        # https://github.com/mindslab-ai/cotatron/blob/master/modules/zoneout.py
        self.bias_ih[hidden_size:2*hidden_size].data.fill_(1.0)
        self.bias_hh[hidden_size:2*hidden_size].data.fill_(1.0)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = super().forward(input, hx)

        if hx is None:
            prev_h = torch.zeros(input.size(0), self.hidden_size)
            prev_c = torch.zeros(input.size(0), self.hidden_size)
        else:
            prev_h, prev_c = hx

        if self.training:
            h = (1. - self.zoneout_rate) * self.dropout(h - prev_h) + prev_h
            c = (1. - self.zoneout_rate) * self.dropout(c - prev_c) + prev_c
        else:
            h = (1. - self.zoneout_rate) * h + self.zoneout_rate * prev_h
            c = (1. - self.zoneout_rate) * c + self.zoneout_rate * prev_c

        return h, c