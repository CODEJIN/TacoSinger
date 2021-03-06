from argparse import Namespace
from hgtk.letter import decompose
import torch
import numpy as np
import pickle, os, hgtk
from typing import Dict

def Decompose(syllable):
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Text_to_Token(text: list, token_dict: dict):
    return np.array([token_dict[x] for x in text], dtype= np.int32)

def Feature_Stack(features: list):
    max_length = max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_length - feature.shape[0]], [0, 0]], constant_values= -10.0) for feature in features],
        axis= 0
        )

    return features

def Token_Stack(tokens: list, token_dict: dict):
    '''
    The length of tokens becomes +1 for padding value of each duration.
    '''    
    max_length = max([len(token) for token in tokens]) + 1    # 1 is for padding '<X>'
    
    tokens = np.stack(
        [np.pad(token, [0, max_length - len(token)], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
        
    return tokens

def Note_Stack(notes: list):
    '''
    The length of notes becomes +1 for padding value of each duration.
    '''    
    max_length = max([len(note) for note in notes]) + 1    # 1 is for padding '<X>'
    
    notes = np.stack(
        [np.pad(note, [0, max_length - len(note)], constant_values= 0) for note in notes],
        axis= 0
        )
        
    return notes

def Duration_Stack(durations: list):
    '''
    The length of durations becomes +1 for padding value of each duration.
    '''
    max_duration = max([np.sum(duration) for duration in durations])
    max_length = max([len(duration) for duration in durations]) + 1    # 1 is for padding duration(max - sum).
    
    durations = np.stack(
        [np.pad(duration, [0, max_length - len(duration)], constant_values= 0) for duration in durations],
        axis= 0
        )
    durations[:, -1] = max_duration - np.sum(durations, axis= 1)   # To fit the time after sample
        
    return durations


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str
        ):
        super(Dataset, self).__init__()
        self.token_dict = token_dict
        self.pattern_path = pattern_path
        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        self.patterns = metadata_dict['File_List']

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))

        texts, notes, durations = [], [], []
        for text, note, duration in zip(pattern_dict['Text'], pattern_dict['Note'], pattern_dict['Duration']):
            if text == '<X>':
                texts.append(text)
                notes.append(note)
                durations.append(duration)
            else:
                texts.extend(Decompose(text))
                notes.extend([note] * 3)
                split_duration = [duration // 3] * 3
                if duration % 3 == 1:
                    split_duration[1] += 1
                elif duration % 3 == 2:
                    split_duration[0] += 1
                    split_duration[1] += 1
                durations.extend(split_duration)
        
        return Text_to_Token(texts, self.token_dict), notes, durations, pattern_dict['Spectrogram']

    def __len__(self):
        return len(self.patterns)


class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: dict,
        pattern_paths: list= ['./Inference_for_Training/Example.txt']
        ):
        super().__init__()
        self.token_dict = token_dict

        self.patterns = []
        for path in pattern_paths:
            music = []
            for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]:
                duration, text, note = line.strip().split('\t')
                music.append((int(duration), text, int(note)))
            durations, texts, notes = zip(*music)
            self.patterns.append((durations, texts, notes, path))

    def __getitem__(self, idx):
        source_durations, source_texts, source_notes, path = self.patterns[idx]
        
        texts, notes, durations = [], [], []
        for text, note, duration in zip(source_texts, source_notes, source_durations):
            if text == '<X>':
                texts.append(text)
                notes.append(note)
                durations.append(duration)
            else:
                texts.extend(Decompose(text))
                notes.extend([note] * 3)
                split_duration = [duration // 3] * 3
                if duration % 3 == 1:
                    split_duration[1] += 1
                elif duration % 3 == 2:
                    split_duration[0] += 1
                    split_duration[1] += 1
                durations.extend(split_duration)
        
        return Text_to_Token(texts, self.token_dict), notes, durations, source_texts, texts

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(self, token_dict: Dict[str, int]):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, notes, durations, features = zip(*batch)
        
        token_lengths = [token.shape[0] + 1 for token in tokens]    # 1 is for padding '<X>'

        sorting = sorted(range(len(token_lengths)), key=lambda x: token_lengths[x], reverse= True)
        tokens = [tokens[index] for index in sorting]
        notes = [notes[index] for index in sorting]
        durations = [durations[index] for index in sorting]
        features = [features[index] for index in sorting]        
        token_lengths = [token_lengths[index] for index in sorting]
        feature_lengths = [feature.shape[0] for feature in features]

        tokens = Token_Stack(tokens, self.token_dict)
        notes = Note_Stack(notes)
        durations = Duration_Stack(durations)
        features = Feature_Stack(features)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        notes = torch.LongTensor(notes)   # [Batch]
        durations = torch.LongTensor(durations)   # [Batch]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_dim, Time]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]

        return tokens, notes, durations, token_lengths, features, feature_lengths

class Inference_Collater:
    def __init__(self, token_dict: Dict[str, int]):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, notes, durations, texts, decomposed_texts = zip(*batch)
        
        token_lengths = [token.shape[0] + 1 for token in tokens]    # 1 is for padding '<X>'

        sorting = sorted(range(len(token_lengths)), key=lambda x: token_lengths[x], reverse= True)
        tokens = [tokens[index] for index in sorting]
        notes = [notes[index] for index in sorting]
        durations = [durations[index] for index in sorting]
        token_lengths = [token_lengths[index] for index in sorting]
        feature_lengths = [sum(duration) for duration in durations]

        texts = [texts[index] for index in sorting]
        decomposed_texts = [decomposed_texts[index] for index in sorting]

        tokens = Token_Stack(tokens, self.token_dict)
        notes = Note_Stack(notes)
        durations = Duration_Stack(durations)
        
        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        notes = torch.LongTensor(notes)   # [Batch]
        durations = torch.LongTensor(durations)   # [Batch]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]

        restoring = sorted(range(len(sorting)), key=lambda x: sorting[x])

        return tokens, notes, durations, token_lengths, feature_lengths, texts, decomposed_texts, restoring