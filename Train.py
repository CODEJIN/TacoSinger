import os
from torch._C import device
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from librosa import griffinlim
from scipy.io import wavfile

from Modules import TacoSinger
from Guided_Attention import Guided_Attention_Loss
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger

from meldataset import spectral_de_normalize_torch
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_Path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Datset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
            }

    def Datset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File
            )
        dev_dataset = Dataset(
            token_dict= token_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            pattern_paths= self.hp.Train.Inference_Pattern_in_Train
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset)))
            logging.info('The number of development patterns = {}.'.format(len(dev_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(token_dict= token_dict)
        inference_collater = Inference_Collater(token_dict= token_dict)

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_dataset,
            sampler= torch.utils.data.DistributedSampler(dev_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(dev_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model = TacoSinger(self.hp).to(self.device)
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss().to(self.device),
            'GAL': Guided_Attention_Loss(),
            }
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= self.hp.Train.Learning_Rate.Base
            )

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model)

    def Train_Step(self, tokens, notes, durations, lengths, features):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            pre_features, post_features, alignments = self.model(
                tokens= tokens,
                notes= notes,
                durations= durations,
                lengths= lengths,
                features= features,
                is_training= True
                )

            loss_dict['Pre'] = self.criterion_dict['MSE'](pre_features, features)
            loss_dict['Post'] = self.criterion_dict['MSE'](post_features, features)
            loss_dict['Guided_Attention'] = self.criterion_dict['GAL'](alignments, lengths, lengths)
            loss_dict['Total'] = loss_dict['Pre'] + loss_dict['Post'] + loss_dict['Guided_Attention']

        self.optimizer.zero_grad()
        self.scaler.scale(loss_dict['Total']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, notes, durations, lengths, features in self.dataloader_dict['Train']:
            self.Train_Step(tokens, notes, durations, lengths, features)

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return


    @torch.no_grad()
    def Evaluation_Step(self, tokens, notes, durations, lengths, features):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)

        pre_features, post_features, alignments = self.model(
            tokens= tokens,
            notes= notes,
            durations= durations,
            lengths= lengths,
            features= features,
            is_training= True
            )

        loss_dict['Pre'] = self.criterion_dict['MSE'](pre_features, features)
        loss_dict['Post'] = self.criterion_dict['MSE'](post_features, features)
        loss_dict['Guided_Attention'] = self.criterion_dict['GAL'](alignments, lengths, lengths)
        loss_dict['Total'] = loss_dict['Pre'] + loss_dict['Post'] + loss_dict['Guided_Attention']

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return pre_features, post_features, alignments

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (tokens, notes, durations, lengths, features) in tqdm(
            enumerate(self.dataloader_dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Dev'].dataset) / self.hp.Train.Batch_Size)
            ):
            pre_features, post_features, alignments = self.Evaluation_Step(tokens, notes, durations, lengths, features)

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_Dict['Evaluation'].add_histogram_model(self.model, 'TacoSinger', self.steps, delete_keywords=[])
        
            image_dict = {
                'Feature/Target': (features[-1].cpu().numpy(), None, 'auto', None),
                'Feature/Pre': (pre_features[-1].cpu().numpy(), None, 'auto', None),
                'Feature/Post': (post_features[-1].cpu().numpy(), None, 'auto', None),
                'Alignment': (alignments[-1].cpu().numpy(), None, 'auto', None)
                }
            self.writer_Dict['Evaluation'].add_image_dict(image_dict, self.steps)

        self.scalar_dict['Evaluation'] = defaultdict(float)

        self.model.train()


    @torch.no_grad()
    def Inference_Step(self, tokens, notes, durations, lengths, texts, decomposed_texts, restoring, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)

        _, post_features, alignments = self.model(
            tokens= tokens,
            notes= notes,
            durations= durations,
            lengths= lengths,
            is_training= False
            )

        post_features = torch.stack([post_features[index] for index in restoring], dim= 0)
        alignments = torch.stack([alignments[index] for index in restoring], dim= 0)
        lengths = torch.stack([lengths[index] for index in restoring], dim= 0)
        texts = [texts[index] for index in restoring]
        decomposed_texts = [decomposed_texts[index] for index in restoring]

        audios = []
        for feature, length in zip(post_features, lengths):
            feature = spectral_de_normalize_torch(feature[:, :length]).cpu().numpy()
            audio = griffinlim(feature)
            audios.append(audio)
        audios = [(audio / np.abs(audio).max() * 32767.5).astype(np.int16) for audio in audios]

        files = []
        for index in range(post_features.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (feature, alignment, text, decomposed_text, audio, file) in enumerate(zip(
            post_features.cpu().numpy(),
            alignments.cpu().numpy(),
            texts,
            decomposed_texts,
            audios,
            files
            )):
            title = 'Text: {}'.format(text if len(text) < 90 else text[:90] + '…')
            new_Figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(feature, aspect='auto', origin='lower')
            plt.title('Feature    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0), rowspan= 2)
            plt.imshow(alignment[:len(decomposed_text) + 2], aspect='auto', origin='lower')
            plt.title('Alignment    {}'.format(title))
            plt.yticks(
                range(len(decomposed_text)),
                list(decomposed_text),
                fontsize = 10
                )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, notes, durations, lengths, texts, decomposed_texts, restoring) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, notes, durations, lengths, texts, decomposed_texts, restoring, start_index= step * batch_size)

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.optimizer.load_state_dict(state_dict['Optimizer'])
        self.scheduler.load_state_dict(state_dict['Scheduler'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))


    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            yaml.dump(self.hp, open(hp_path, 'w'))

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    argParser.add_argument('-p', '--port', default= 54321, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl',
            dist_url= 'tcp://127.0.0.1:{}'.format(args.port)
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()