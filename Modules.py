from argparse import Namespace
import torch
from dca_attention import Attention
from zoneout import ZoneoutLSTMCell

class TacoSinger(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        self.decoder = Decoder(self.hp)
        self.postnet = Postnet(self.hp)

    def forward(
        self,
        tokens: torch.LongTensor,
        notes: torch.LongTensor,
        durations: torch.LongTensor,
        lengths: torch.LongTensor,
        features: torch.FloatTensor= None,
        is_training: bool= False
        ):
        encodings = self.encoder(tokens, notes, durations, lengths)

        pre_features, alignments = self.decoder(
            encodings= encodings,
            lengths= lengths,
            features= features,
            is_training= is_training
            )

        post_features = self.postnet(pre_features)
        
        masks = self.Mask_Generate(lengths)
        pre_features.data.masked_fill_(masks.unsqueeze(1), -10.0)
        post_features.data.masked_fill_(masks.unsqueeze(1), -10.0)

        return pre_features, post_features, alignments

    def Mask_Generate(self, lengths, max_lengths= None):
        '''
        lengths: [Batch]
        '''
        sequence = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device)
        return sequence >= lengths[:, None]    # [Batch, Time]


class Encoder(torch.nn.Module): 
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        assert self.hp.Encoder.LSTM.Size % 2 == 0, 'The LSTM size of text encoder must be a even number.'

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Token_Embedding_Size,
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Max_Note,
            embedding_dim= self.hp.Encoder.Note_Embedding_Size,
            )
        torch.nn.init.xavier_uniform_(self.token_embedding.weight)
        torch.nn.init.xavier_uniform_(self.note_embedding.weight)
        
        previous_channels = self.hp.Encoder.Token_Embedding_Size + self.hp.Encoder.Note_Embedding_Size
        self.conv = torch.nn.Sequential()
        for index, (kernel_size, channels) in enumerate(zip(
            self.hp.Encoder.Conv.Kernel_Size,
            self.hp.Encoder.Conv.Channels
            )):
            self.conv.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_channels,
                out_channels= channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.conv.add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= channels
                ))
            self.conv.add_module('ReLU_{}'.format(index), torch.nn.ReLU(inplace= False))
            self.conv.add_module('Dropout_{}'.format(index), torch.nn.Dropout(
                p= self.hp.Encoder.Conv.Dropout,
                inplace= True
                ))
            previous_channels = channels

        self.lstm = torch.nn.LSTM(
            input_size= previous_channels,
            hidden_size= self.hp.Encoder.LSTM.Size // 2,
            num_layers= self.hp.Encoder.LSTM.Stacks,
            batch_first= True,
            bidirectional= True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        durations: torch.Tensor,
        lengths: torch.Tensor
        ):
        '''
        tokens: [Batch, Time]
        notes: [Batch, Time]
        durations: [Batch, Time]
        lengths: [Batch]
        '''
        x = torch.cat([
            self.token_embedding(tokens),
            self.note_embedding(notes)
            ], dim= 2).transpose(2, 1)     # [Batch, Dim, Time]
        x = torch.stack([
            x.repeat_interleave(duration, dim= 1)
            for x, duration in zip(x, durations)
            ], dim= 0)  # [Batch, Dim, Time]

        x = self.conv(x)    # [Batch, Dim, Time]
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1),
            lengths.cpu().numpy(),
            batch_first= True
            )
        x, _ = self.lstm(x)    # [Batch, Time, Dim]
        x = torch.nn.utils.rnn.pad_packed_sequence(sequence= x, batch_first= True)[0].transpose(2, 1)   # [Batch, Dim, Time]

        return x

class Decoder(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters
        self.feature_size = self.hp.Sound.N_FFT // 2 + 1

        self.go_frame = torch.nn.Parameter(
            data= torch.randn(1, self.feature_size, 1),
            requires_grad=True
            )

        self.prenet = Prenet(
            in_features= self.feature_size,
            layer_sizes= self.hp.Decoder.Prenet.Sizes,
            dropout_rate= self.hp.Decoder.Prenet.Dropout_Rate
            )
        
        self.pre_lstm = ZoneoutLSTMCell(
            input_size= self.hp.Decoder.Prenet.Sizes[-1] + self.hp.Encoder.LSTM.Size,   # encoding size == previous context size
            hidden_size= self.hp.Decoder.Pre_LSTM.Size,
            bias= True,
            zoneout_rate= self.hp.Decoder.Pre_LSTM.Zoneout_Rate
            )

        self.attention = Attention(
            attn_rnn_dim= self.hp.Decoder.Pre_LSTM.Size,
            attn_dim= self.hp.Decoder.Attention.Channels,
            static_channels= self.hp.Decoder.Attention.Static.Channels,
            static_kernel_size= self.hp.Decoder.Attention.Static.Kernel_Size,
            dynamic_channels= self.hp.Decoder.Attention.Dynamic.Channels,
            dynamic_kernel_size= self.hp.Decoder.Attention.Dynamic.Kernel_Size,
            causal_n= self.hp.Decoder.Attention.Causal.Kernel_Size,
            causal_alpha= self.hp.Decoder.Attention.Causal.Alpha,
            causal_beta= self.hp.Decoder.Attention.Causal.Beta
            )

        self.post_lstm = ZoneoutLSTMCell(
            input_size= self.hp.Decoder.Pre_LSTM.Size + self.hp.Encoder.LSTM.Size,
            hidden_size= self.hp.Decoder.Post_LSTM.Size,
            bias= True,
            zoneout_rate= self.hp.Decoder.Post_LSTM.Zoneout_Rate
            )

        self.projection = Linear(
            in_features= self.hp.Decoder.Post_LSTM.Size + self.hp.Encoder.LSTM.Size,
            out_features= self.feature_size,
            bias= True
            )
        
    def forward(self, encodings, lengths, features= None, is_training= False):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        lengths: [Batch]
        features: [Batch, Feature_d, Dec_t] or None(inference)
        is_training: bool
        '''
        if is_training:
            return self.Train(encodings, lengths, features)
        else:
            return self.Inference(encodings, lengths)

    def Train(self, encodings, lengths, features):
        features = torch.cat([self.Get_Initial_Features(encodings), features[:, :, :-1]], dim= 2)
        features = self.prenet(features.transpose(2, 1))  # [Batch, Feature_t, Prenet_dim]

        encodings = encodings.transpose(2, 1)   # [Batch, Enc_t, Enc_dim]
        encoding_masks = self.Mask_Generate(lengths)

        pre_lstm_hidden, pre_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Pre_LSTM.Size
            )
        post_lstm_hidden, post_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Post_LSTM.Size
            )
        contexts, alignments = self.Get_Attention_Initial_States(
            memories= encodings
            )
        
        features_list, alignments_list = [], []
        for step in range(features.size(1)):
            x = torch.cat([features[:, step], contexts], dim= 1)
            
            pre_lstm_hidden, pre_lstm_cell = self.pre_lstm(
                x,  # contexts_t-1
                (pre_lstm_hidden, pre_lstm_cell)
                )

            contexts, alignments = self.attention(
                attn_hidden= pre_lstm_hidden,
                memory= encodings,
                prev_attn= alignments,
                mask= encoding_masks
                )

            post_lstm_hidden, post_lstm_cell = self.post_lstm(
                torch.cat([pre_lstm_hidden, contexts], dim= 1),  # contexts_t
                (post_lstm_hidden, post_lstm_cell)
                )

            decodings = torch.cat([post_lstm_hidden, contexts], dim= 1)

            projections = self.projection(decodings)

            features_list.append(projections)
            alignments_list.append(alignments)

        features = torch.stack(features_list, dim= 2)  # [Batch, Feature_dim, Feature_t]
        alignments = torch.stack(alignments_list, dim= 2)  # [Batch, Key_t, Query_t]

        return features, alignments

    def Inference(self, encodings, lengths):
        features = self.Get_Initial_Features(encodings).transpose(2, 1) # [Batch, 1, Feature_dim]
        encodings = encodings.transpose(2, 1)   # [Batch, Enc_t, Enc_dim]
        encoding_masks = self.Mask_Generate(lengths)

        pre_lstm_hidden, pre_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Pre_LSTM.Size
            )
        post_lstm_hidden, post_lstm_cell = self.Get_LSTM_Intial_States(
            reference= encodings,
            cell_size= self.hp.Decoder.Post_LSTM.Size
            )
        contexts, alignments = self.Get_Attention_Initial_States(
            memories= encodings
            )
        
        features_list, alignments_list = [features], []

        step = 0
        while True:
            x = torch.cat([
                self.prenet(features_list[-1].squeeze(1)),
                contexts
                ], dim= 1)

            pre_lstm_hidden, pre_lstm_cell = self.pre_lstm(
                x,  # contexts_t-1
                (pre_lstm_hidden, pre_lstm_cell)
                )

            contexts, alignments = self.attention(
                attn_hidden= pre_lstm_hidden,
                memory= encodings,
                prev_attn= alignments,
                mask= encoding_masks
                )

            post_lstm_hidden, post_lstm_cell = self.post_lstm(
                torch.cat([pre_lstm_hidden, contexts], dim= 1),  # contexts_t
                (post_lstm_hidden, post_lstm_cell)
                )

            decodings = torch.cat([post_lstm_hidden, contexts], dim= 1)

            projections = self.projection(decodings)
            
            features_list.append(projections)
            alignments_list.append(alignments)

            if step >= lengths.max() - 1:
                break
            step += 1

        features = torch.stack(features_list[1:], dim= 2)  # [Batch, Feature_dim, Feature_t]
        alignments = torch.stack(alignments_list, dim= 2)  # [Batch, Key_t, Query_t]

        return features, alignments


    def Get_Initial_Features(self, reference):
        return self.go_frame.expand(reference.size(0), self.go_frame.size(1), 1)

    def Get_LSTM_Intial_States(self, reference, cell_size):
        hiddens = reference.new_zeros(
            size= (reference.size(0), cell_size)
            )
        cells = reference.new_zeros(
            size= (reference.size(0), cell_size)
            )

        return hiddens, cells

    def Get_Attention_Initial_States(self, memories):
        '''
        memories: [Batch, Enc_t, Enc_dim]
        '''
        contexts = memories.new_zeros(
            size= (memories.size(0), memories.size(2))
            )
        alignments = memories.new_zeros(
            size= (memories.size(0), memories.size(1))
            )
        alignments[:, 0] = 1.0   # (Q0, M0) is 1.0
        
        return contexts, alignments

    def Mask_Generate(self, lengths, max_lengths= None):
        '''
        lengths: [Batch]
        '''
        sequence = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device)
        return sequence >= lengths[:, None]    # [Batch, Time]

class Prenet(torch.nn.Sequential):
    def __init__(self, in_features, layer_sizes, dropout_rate= 0.5):
        super().__init__()
        previous_features = in_features
        for index, size in enumerate(layer_sizes):
            self.add_module('Linear_{}'.format(index), Linear(
                in_features= previous_features,
                out_features= size,
                bias= False,
                w_init_gain= 'relu'
                ))
            self.add_module('ReLU_{}'.format(index), torch.nn.ReLU())
            self.add_module('Dropout_{}'.format(index), torch.nn.Dropout(p= dropout_rate, inplace= True))
            previous_features = size

    def forward(self, x):
        for name, module in self.named_modules():
            if name.startswith('Dropout'):  module.train()  # Prenet's dropout is always on.
        return super().forward(x)

class Postnet(torch.nn.Sequential):
    def __init__(self, hyper_parameters: Namespace):
        super(Postnet, self).__init__()
        self.hp = hyper_parameters
        self.feature_size = self.hp.Sound.N_FFT // 2 + 1

        previous_channels = self.feature_size
        for index, (kernel_size, channels) in enumerate(zip(
            self.hp.Postnet.Kernel_Size + [5],
            self.hp.Postnet.Channels + [self.feature_size],
            )):
            self.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_channels,
                out_channels= channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2,
                w_init_gain= 'tanh' if index < len(self.hp.Postnet.Kernel_Size) else 'linear'
                ))
            self.add_module('BatchNorm_{}'.format(index), torch.nn.BatchNorm1d(
                num_features= channels
                ))
            if index < len(self.hp.Postnet.Kernel_Size):
                self.add_module('Tanh_{}'.format(index), torch.nn.Tanh())
            self.add_module('Dropout_{}'.format(index), torch.nn.Dropout(
                p= self.hp.Postnet.Dropout_Rate
                ))
            previous_channels = channels

    def forward(self, x):
        return super().forward(x) + x


class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias= True, w_init_gain='linear'):
        self.w_init_gain = w_init_gain
        super(Linear, self).__init__(
            in_features= in_features,
            out_features= out_features,
            bias= bias
            )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)
