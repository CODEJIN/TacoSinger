import torch
from math import sqrt

# Currently, this code is using transpose function several times by 'channels_last'.
# This is to sustain the compatibility with Nvidia Location sensitive attention module.
# If you want to apply this code to other model, consider and modify that.

class Location_Sensitive_Stepwise_Monotonic_Attention(torch.nn.Module):
    def __init__(
        self,
        attention_rnn_channels,
        memory_size,
        attention_size,
        attention_location_channels,
        attention_location_kernel_size,
        sigmoid_noise= 2.0,
        score_bias= False,
        normalize= False,
        channels_last= False,
        **kwargs
        ):
        super(Location_Sensitive_Stepwise_Monotonic_Attention, self).__init__()
        self.sigmoid_noise = sigmoid_noise
        self.normalize = normalize
        self.channels_last = channels_last

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Query'] = ConvNorm(
            in_channels= attention_rnn_channels,
            out_channels= attention_size,
            kernel_size= 1,
            bias= False,
            w_init_gain= 'tanh'
            )
        self.layer_Dict['Memory'] = ConvNorm(
            in_channels= memory_size,
            out_channels= attention_size,
            kernel_size= 1,
            bias= False,
            w_init_gain= 'tanh'
            )

        self.layer_Dict['Location'] = torch.nn.Sequential()
        self.layer_Dict['Location'].add_module('Conv', ConvNorm(
            in_channels= 2,
            out_channels= attention_location_channels,
            kernel_size= attention_location_kernel_size,
            padding= (attention_location_kernel_size - 1) // 2,
            bias= False
            ))
        self.layer_Dict['Location'].add_module('Conv1x1', ConvNorm(
            in_channels= attention_location_channels,
            out_channels= attention_size,
            kernel_size= 1,
            bias= False,
            w_init_gain= 'tanh'
            ))
        
        self.layer_Dict['Score'] = torch.nn.Sequential()
        self.layer_Dict['Score'].add_module('Tanh', torch.nn.Tanh())
        self.layer_Dict['Score'].add_module('Conv', ConvNorm(
            in_channels= attention_size,
            out_channels= 1,
            kernel_size= 1,
            bias= score_bias
            ))

        if normalize:
            torch.nn.utils.weight_norm(self.layer_Dict['Score'])

    def forward(self, queries, memories, processed_memories, previous_alignments, cumulated_alignments, masks= None):
        '''
        queries: [Batch, Att_RNN_dim]
        memories: [Batch, Enc_dim, Memory_t] or [Batch, Memory_t, Enc_dim] (when channels_list is True)
        processed_memories: [Batch, Att_dim, Memory_t] or [Batch, Memory_t, Att_dim] (when channels_last is True)
        attention_weights_cats: [Batch, 2, Memory_t]
        mask: None or [Batch, Memory_t]
        '''
        if self.channels_last:
            memories = memories.transpose(2, 1)
            processed_memories = processed_memories.transpose(2, 1)

        scores = self.Calc_Score(   # [Batch, Mem_t]
            queries= queries,
            memories= processed_memories,
            attention_weights_cats= torch.cat([previous_alignments.unsqueeze(1), cumulated_alignments.unsqueeze(1)], dim= 1)
            )
        contexts, alignments = self.Apply_Score(scores, memories, previous_alignments, masks)   # [Batch, Att_dim], [Batch, Att_dim]

        return contexts, alignments

    def Calc_Score(self, queries, memories, attention_weights_cats):
        queries = self.layer_Dict['Query'](queries.unsqueeze(2)) # [Batch, Att_dim, 1]
        locations = self.layer_Dict['Location'](attention_weights_cats)   # [Batch, Att_dim, Mem_t]
        
        return self.layer_Dict['Score'](queries + memories + locations).squeeze(1)

    def Apply_Score(self, scores, memories, previous_alignments, masks= None):
        previous_alignments = previous_alignments.unsqueeze(1)

        if self.sigmoid_noise > 0.0:
            scores += self.sigmoid_noise * torch.randn_like(scores)

        if not masks is None:
            scores.data.masked_fill_(masks, -torch.finfo(scores.dtype).max)

        p_choose_i = torch.sigmoid(scores).unsqueeze(1)  # [Batch, 1, Mem_t]
        pad = torch.zeros(p_choose_i.size(0), 1, 1).to(device= p_choose_i.device, dtype= p_choose_i.dtype)     # [Batch, 1, 1]

        alignments = previous_alignments * p_choose_i + torch.cat(
            [pad, previous_alignments[..., :-1] * (1.0 - p_choose_i[..., :-1])],
            dim= -1
            )   # [Batch, 1, Mem_t]

        contexts = alignments @ memories.transpose(2, 1)   # [Batch, 1, Att_dim]

        return contexts.squeeze(1), alignments.squeeze(1)
    
    def Get_Processed_Memory(self, memories):
        if self.channels_last:
            return self.layer_Dict['Memory'](memories.transpose(2, 1)).transpose(2, 1)

        return self.layer_Dict['Memory'](memories)

    def Get_Initial_Alignment(self, memories):
        return torch.nn.functional.one_hot(
            memories.new_zeros(memories.size(0)).long(),
            num_classes= memories.size(1) if self.channels_last else memories.size(2)
            ).to(dtype= memories.dtype)


class ConvNorm(torch.nn.Conv1d):
    def __init__(self, w_init_gain='linear', *args, **kwargs):
        super(ConvNorm, self).__init__(*args, **kwargs)
        torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(w_init_gain))