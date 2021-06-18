from argparse import Namespace
import torch
import math
from LSSMA import Location_Sensitive_Stepwise_Monotonic_Attention as Attention

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
        token_lengths: torch.LongTensor,
        feature_lengths: torch.LongTensor,
        features: torch.FloatTensor= None,
        is_training: bool= False
        ):
        encodings = self.encoder(tokens, notes, durations, token_lengths)

        pre_features, alignments = self.decoder(
            encodings= encodings,
            encoding_lengths= token_lengths,
            features= features,
            feature_lengths= feature_lengths,
            is_training= is_training
            )

        post_features = self.postnet(pre_features)
        
        masks = self.Mask_Generate(feature_lengths)
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

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Max_Note,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.positional_encoding = Positional_Encoding(
            max_position= self.hp.Max_Duration * 2,
            embedding_size= self.hp.Encoder.Size,
            dropout_rate= self.hp.Encoder.Positional_Encoding.Dropout_Rate
            )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer= torch.nn.TransformerEncoderLayer(
                d_model= self.hp.Encoder.Size,
                nhead= self.hp.Encoder.Transformer.Head,
                dim_feedforward= self.hp.Encoder.Size * 4,
                dropout= self.hp.Encoder.Transformer.Dropout_Rate
                ),
            num_layers= self.hp.Encoder.Transformer.Num_Layers,
            norm= torch.nn.LayerNorm(
                normalized_shape= self.hp.Encoder.Size
                )
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
        '''
        x = self.token_embedding(tokens) + self.note_embedding(notes)   # [Batch, Time, Dim]
        x = self.positional_encoding(x.permute(0, 2, 1)) # [Batch, Dim, Time]
        x = self.transformer(x.permute(2, 0, 1))    # [Time, Batch, Emb_dim]

        return x.permute(1, 2, 0)   # [Batch, Emb_dim, Time]

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
        
        self.pre_lstm = torch.nn.LSTMCell(
            input_size= self.hp.Decoder.Prenet.Sizes[-1] + self.hp.Encoder.Size,   # encoding size == previous context size
            hidden_size= self.hp.Decoder.Pre_LSTM.Size,
            bias= True
            )
        self.pre_lstm_dropout = torch.nn.Dropout(
            p= self.hp.Decoder.Pre_LSTM.Dropout_Rate
            )

        self.attention = Attention(
            attention_rnn_channels= self.hp.Decoder.Pre_LSTM.Size,
            memory_size= self.hp.Encoder.Size,
            attention_size= self.hp.Decoder.Attention.Channels,
            attention_location_channels= self.hp.Decoder.Attention.Conv.Channels,
            attention_location_kernel_size= self.hp.Decoder.Attention.Conv.Kernel_Size,
            sigmoid_noise= self.hp.Decoder.Attention.Sigmoid_Noise,
            normalize= self.hp.Decoder.Attention.Normalize,
            channels_last= True
            )


        self.post_lstm = torch.nn.LSTMCell(
            input_size= self.hp.Decoder.Pre_LSTM.Size + self.hp.Encoder.Size,
            hidden_size= self.hp.Decoder.Post_LSTM.Size,
            bias= True
            )
        self.post_lstm_dropout = torch.nn.Dropout(
            p= self.hp.Decoder.Post_LSTM.Dropout_Rate
            )

        self.projection = Linear(
            in_features= self.hp.Decoder.Post_LSTM.Size + self.hp.Encoder.Size,
            out_features= self.feature_size,
            bias= True
            )
        
    def forward(self, encodings, encoding_lengths, features= None, feature_lengths= None, is_training= False):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        encoding_lengths: [Batch]
        features: [Batch, Feature_d, Dec_t] or None(inference)
        is_training: bool
        '''
        if is_training:
            return self.Train(encodings, encoding_lengths, features)
        else:
            return self.Inference(encodings, encoding_lengths, feature_lengths)

    def Train(self, encodings, encoding_lengths, features):
        features = torch.cat([self.Get_Initial_Features(encodings), features[:, :, :-1]], dim= 2)
        features = self.prenet(features.transpose(2, 1))  # [Batch, Feature_t, Prenet_dim]

        encodings = encodings.transpose(2, 1)   # [Batch, Enc_t, Enc_dim]
        encoding_masks = self.Mask_Generate(encoding_lengths)

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
        cumulated_alignments = alignments
        processed_memories = self.attention.Get_Processed_Memory(encodings)
        
        features_list, alignments_list = [], []
        for step in range(features.size(1)):
            x = torch.cat([features[:, step], contexts], dim= 1)
            
            pre_lstm_hidden, pre_lstm_cell = self.pre_lstm(
                x,  # contexts_t-1
                (pre_lstm_hidden, pre_lstm_cell)
                )
            pre_lstm_hidden = self.pre_lstm_dropout(pre_lstm_hidden)

            contexts, alignments = self.attention(
                queries= pre_lstm_hidden,
                memories= encodings,
                processed_memories= processed_memories,
                previous_alignments= alignments,
                cumulated_alignments= cumulated_alignments,
                masks= encoding_masks
                )
            cumulated_alignments = cumulated_alignments + alignments

            post_lstm_hidden, post_lstm_cell = self.post_lstm(
                torch.cat([pre_lstm_hidden, contexts], dim= 1),  # contexts_t
                (post_lstm_hidden, post_lstm_cell)
                )
            post_lstm_hidden = self.post_lstm_dropout(post_lstm_hidden)

            decodings = torch.cat([post_lstm_hidden, contexts], dim= 1)

            projections = self.projection(decodings)

            features_list.append(projections)
            alignments_list.append(alignments)

        features = torch.stack(features_list, dim= 2)  # [Batch, Feature_dim, Feature_t]
        alignments = torch.stack(alignments_list, dim= 2)  # [Batch, Key_t, Query_t]

        return features, alignments

    def Inference(self, encodings, encoding_lengths, feature_lengths):
        features = self.Get_Initial_Features(encodings).transpose(2, 1) # [Batch, 1, Feature_dim]
        encodings = encodings.transpose(2, 1)   # [Batch, Enc_t, Enc_dim]
        encoding_masks = self.Mask_Generate(encoding_lengths)

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
        cumulated_alignments = alignments
        processed_memories = self.attention.Get_Processed_Memory(encodings)
        
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
                queries= pre_lstm_hidden,
                memories= encodings,
                processed_memories= processed_memories,
                previous_alignments= alignments,
                cumulated_alignments= cumulated_alignments,
                masks= encoding_masks
                )
            cumulated_alignments = cumulated_alignments + alignments

            post_lstm_hidden, post_lstm_cell = self.post_lstm(
                torch.cat([pre_lstm_hidden, contexts], dim= 1),  # contexts_t
                (post_lstm_hidden, post_lstm_cell)
                )

            decodings = torch.cat([post_lstm_hidden, contexts], dim= 1)

            projections = self.projection(decodings)
            
            features_list.append(projections)
            alignments_list.append(alignments)

            if step >= feature_lengths.max() - 1:
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

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
class Positional_Encoding(torch.nn.Module):
    def __init__(
        self,
        max_position: int,
        embedding_size: int,
        dropout_rate: float,
        sin_cos_change: bool= False
        ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        pe = torch.zeros(max_position, embedding_size)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        if sin_cos_change:
            pe[:, 0::2] = torch.cos(position * div_term)
            pe[:, 1::2] = torch.sin(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(2, 1)
        self.register_buffer('pe', pe)

        self.alpha = torch.nn.Parameter(
            data= torch.ones(1),
            requires_grad= True
            )

    def forward(self, x):
        '''
        x: [Batch, Dim, Length]
        '''
        x = x + self.alpha * self.pe[:, :, :x.size(2)]
        x = self.dropout(x)

        return x
