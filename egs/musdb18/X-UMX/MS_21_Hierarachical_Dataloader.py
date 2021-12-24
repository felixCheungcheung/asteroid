# the sample code in pytorch example combines dataloader and datasets in the same part.
# whereas in asteroid x-umx code, they are separated, 
# it is good to separated when there are more than one dataset are trained on.

from pathlib import Path
import torch.nn.functional as F
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf
import os
import pandas as pd
import json
from collections.abc import MutableMapping
import pandas as pd
# import musedb

class MS_21Dataset(torch.utils.data.Dataset):
    """MS_21 music separation dataset

    The dataset consists of 150 full lengths music tracks (~10h duration) of
    different genres along with their raw multitracks:
    

    This dataset asssumes music raw multi-tracks in (sub)folders where each folder
    has a various number of sources. 
    A linear mix is performed on the fly by summing up the sources according to 
    the grouping information in the .csv file.
    In order to be compatible to MUSDB_18 dataset, one can utilize the grouping information
    to generate the traditional four stems:
        'drums', 'vocals', 'bass', 'other'
    

    Folder Structure:
        >>> #train/1/lead_vocals.wav ------------|
        >>> #train/1/backing_vocals.wav ---------|
        >>> #train/1/drums.wav ---------------+--> input (mix),
        >>> #train/1/bass.wav -------------------|
        >>> #train/1/accordin.wav ---------------|
        >>> #train/1/bell.wav -------------------/

        >>> #train/1/lead_vocals.wav ------------> output[target]

    Args:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            that composes the mixture.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        targets (list or None, optional): List of source names to be used as
            targets. If None, a dict with the 4 stems is returned.
             If e.g [`vocals`, `drums`], a tensor with stacked `vocals` and
             `drums` is returned instead of a dict. Defaults to None.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.

    Attributes:
        root (str): Root path of dataset
        sources (:obj:`list` of :obj:`str`, optional): List of source names.
            Defaults to MUSDB18 4 stem scenario: `vocals`, `drums`, `bass`, `other`.
        suffix (str, optional): Filename suffix, defaults to `.wav`.
        split (str, optional): Dataset subfolder, defaults to `train`.
        subset (:obj:`list` of :obj:`str`, optional): Selects a specific of
            list of tracks to be loaded, defaults to `None` (loads all tracks).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
        random_track_mix boolean: enables mixing of random sources from
            different tracks to assemble mix.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
        tracks (:obj:`list` of :obj:`Dict`): List of track metadata

    References
        "The 2018 Signal Separation Evaluation Campaign" Stoter et al. 2018.
    """

    dataset_name = "MS_21"

    def __init__(
        self,
        root,
        csv_file_path,
        grouping_info = {'drums':['Drum_Kick','Drum_Snare','Drum_HiHat','Drum_Cymbals','Drum_Overheads','Drum_Tom','Drum_Room','Percussion'
],'vocals':['Lead_Vocal','Backing_Vocal'],'bass':['Bass'],'other':['Acoustic_Guitar','Electric_Guitar','Piano','Electric_Piano','Brass','String','WoodWind','Other'
]}, # default traditional four stems grouping style
        sources=["vocals", "bass", "drums", "other"],
        targets=None,
        suffix=".wav",
        split="train",
        subset=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        nb_level = 3,
        multi_level = True

    ):

        self.root = Path(root).expanduser()
        self.csv_info = pd.read_csv(csv_file_path)
        with open('hierarchy.json') as json_file:
            self.hierarchy_info = json.load(json_file)
        self.grouping_info = grouping_info
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())
        self.nb_level = nb_level
        self.multi_level = multi_level

        #print(self.tracks)
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        # self.__getitem__(index = 1)

    def retrieve_hie(target, d: MutableMapping, sep='.'): 

        # retrive the instruments names which construct the corresponding level
        # e.g. child_level: Lead_Vocal; parent_level(vocal): Lead_Vocal, Backing_Vocal; Grandparent_level(vocal, wind):Lead_Vocal, Backing_Vocal,Woodwind, Brass
        [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
        hie_fla = flat_dict
        child_level = [target]
        parent_level = []
        grandparent_level = []
        for keys,values in enumerate(hie_fla):
            if target in hie_fla[values][:]: # search every leaf node (child level)
                grandparent_level_name = values.split('.')[-2]
                grandparent_level =list( d['mix'][grandparent_level_name].values())
                grandparent_level = [item for sublist in grandparent_level for item in sublist]
                parent_level_name = values.split('.')[-1]
                parent_level = hie_fla[values][:]

        return child_level, parent_level, grandparent_level

    def source_stack(audio_sources, target):
        # target should be in child level
        res = torch.stack([wav for src, wav in audio_sources.items() if src in target], dim=0)
        return res

    def source_mix(audio_sources, level):
        
        audio_mix = torch.stack(list(audio_sources[i]) for i in level).sum(0)
        return audio_mix

    def __getitem__(self, index):
        # create a dict for storing stem grouping rule
        
        
        
        # assemble the mixture of target and interferers
        audio_sources = {}

        # get track_id
        track_id = index // self.samples_per_track
        
        
        # print("min duration = ", self.tracks[track_id]["min_duration"])
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # create sources based on multitracks
        # for source in self.sources:
        # optionally select a random track for each source
        if self.random_track_mix:
            # load a different track
            track_id = random.choice(range(len(self.tracks)))
            if self.random_segments:
                start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

        # loads the full track duration
        start_sample = int(start * self.sample_rate)
        # check if dur is none
        if self.segment:
            # stop in soundfile is calc in samples, not seconds
            stop_sample = start_sample + int(self.segment * self.sample_rate)
        else:
            # set to None for reading complete file
            stop_sample = None

        # load actual audio
#             audio, _ = sf.read(
#                 Path(self.tracks[track_id]["path"] / source).with_suffix(self.suffix),
#                 always_2d=True,
#                 start=start_sample,
#                 stop=stop_sample,
#             )
        
        track_path = self.tracks[track_id]['path']
        print("Track_Title = ",track_path)
        # load multitracks and be ready to do linear mix
        max_len = []
        for i in self.grouping_info:
            # print(i) # get source names
            stem_tracks = []
            # get all instrument name within one stem
            for j in self.grouping_info[i]:
                # print(j)
                # get all multitrack filename within one instrument
                # get the corresponding song:
                track_title = str(track_path).split('\\')[-1]
                track_title = track_title.replace('_',' ')
                
                track_df = self.csv_info.loc[self.csv_info['Music_Title']==track_title]
                # print(track_df)
                if track_df.empty==True:
                    print("This song doesn't exist in csv file", track_title)
                    
                    
                    
                temp = track_df[j].tolist()[0]
                if temp != '[]':
                    for m in temp.strip('[]').split(', '):
                        # print(m.strip(''))
                        stem_tracks.append(m.strip('\''))
                
            # apply linear mix within one source (stem) later can intergrate with data augmentation
            # first load each multitrack
            source_multitrack = {}
            max_len_source = 0
            for k in stem_tracks:
                audio,_ = sf.read(
                    Path(self.tracks[track_id]['path'] / k),
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
                )
                # convert to torch tensor
                audio = torch.tensor(audio.T, dtype=torch.float)
                
                if list(audio.shape)[0] == 1:
                    audio = audio.repeat(2,1)
                    
                # apply multitrack-wise augmentations
                # audio = self.multitrack_augmentation(audio)
                if max_len_source <= audio.size(dim=1):
                    max_len_source = audio.size(dim=1)
                source_multitrack[k] = audio
                
            # zero-padding to the maximum length of tensor
            max_len.append(max_len_source)

            # print("Max_length = ",max_len_source)
            for key, value in source_multitrack.items():
                if value.size(dim=1)==max_len_source:
                    continue
                else:
                    # print("before zero-padding, value.size = ",value.size(dim=1))
                    target = torch.zeros(2,max_len_source)
                    source_len = value.size(dim=1)
                    target[:,:source_len] = value
                    source_multitrack[key] = target
                    # print("after padding, len = ", source_multitrack[key].size(dim=1))


            

            # apply linear mix over all multitracks within one source index=0
            source_mix = torch.stack(list(source_multitrack.values())).sum(0)
            audio_sources[i] = source_mix
            # apply source-wise augmentations
            # source_mix = self.source_augmentations(source_mix)

        # zero-padding to the maximum length of tensor, between different sources
        

        
        for key, value in audio_sources.items():
            if value.size(dim=1)==max(max_len):
                continue
            else:
                # print("before zero-padding, value.size = ",value.size(dim=1))
                target = torch.zeros(2,max(max_len))
                source_len = value.size(dim=1)
                target[:,:source_len] = value
                audio_sources[key] = target
                # print("after padding, len = ", source_multitrack[key].size(dim=1))
        
        # define that audio_mix is a dict of mixture on three levels
        # through keys one can access all the paths
        # e.g. audio_mix["superlevel_wind_vocal"]["vocal"]["Lead_Vocal"] you can get Lead_Vocal
        # audio_mix["superlevel_wind_vocal"]["vocal"] you can get all vocal,
        # audio_mix["superlevel_wind_vocal"] you can get the mixture of wind and vocal

        if self.multi_level:
            # audio mix = a sequence of mixtures from each layer from root to leaf
            # audio sources = a sequence of sources from root-1 to leaf

            # retrieve the hierarchical structure which contains self.targets e.g. Lead_Vocal
            # hierarchical structure means a path starting from the self.targets
            # e.g. Lead_Vocal - vocal(Lead_Vocal, Backing_Vocal) - wind&vocal(Woodwind, Brass, Lead_Vocal, Backing_Vocal)

            # retrieve_hie returns three lists of instrument names
            child_level, parent_level, grandparent_level = retrieve_hie(self.targets, self.hierarchy_info) 

            # ground truth audio_gt is the target source of the current level
            audio_gt = {}
            audio_gt['child'] = source_stack(audio_sources, child_level) # target is only one
            audio_gt['parent'] = source_mix(audio_sources, parent_level)
            audio_gt['grandparent'] = source_mix(audio_sources, grandparent_level)
            
            # audio_mix: a mixture over the sources from one level higher,
            audio_mix = {}
            audio_mix['child'] = source_mix(audio_sources, parent_level)
            audio_mix['parent'] = source_mix(audio_sources, grandparent_level)
            audio_mix['grandparent'] = torch.stack(list(audio_sources.values())).sum(0)
            

        
        
        return audio_mix, audio_gt

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        """load tracks that contain all the required sources tracks"""
        p = Path(self.root, self.split) # train and test folder
        # p = Path(self.root)
        
        for track_path in tqdm.tqdm(p.iterdir()):
            # print(track_path)
            if track_path.is_dir():
                if self.subset and track_path.stem not in self.subset:
                    # skip this track
                    continue

                
                # source_paths = [track_path / (s + self.suffix) for s in self.sources] # 固定命名
                
                multitrack_paths = []
                for s in os.listdir(track_path):
                    if s.split('.')[-1]=='wav' and s.split('.')[0]!='':
                        multitrack_paths.append(track_path / s )
                

                if not all(sp.exists() for sp in multitrack_paths):
                    print("Exclude track due to non-existing multitrack file", track_path)
                    continue
                
                sources = []
                for i in self.grouping_info:
                    # print(i) # get source names
                    stem_tracks = []
                    # get all instrument name within one stem
                    for j in self.grouping_info[i]:
                        # print(j)
                        # get all multitrack filename within one instrument
                        # get the corresponding song:
                        track_title = str(track_path).split('\\')[-1]
                        track_title = track_title.replace('_',' ')
                        
                        track_df = self.csv_info.loc[self.csv_info['Music_Title']==track_title]
                        # print(track_df)
                        if track_df.empty==True:
                            print("This song doesn't exist in csv file", track_title)

                        temp = track_df[j].tolist()[0]
                        if temp != '[]':
                            for m in temp.strip('[]').split(', '):
                                # print(m.strip(''))
                                stem_tracks.append(m.strip('\''))
                    sources.append(len(stem_tracks))
                if 0 in sources:
                    source_name_list = ['drums','vocals','bass','other']
                    
                    print("Exclude track due to non-existing sources file", track_path, " missing ", source_name_list[sources.index(0)])
                    continue


                # get metadata
                infos = list(map(sf.info, multitrack_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to different sample rate ", track_path)
                    continue

                if self.segment is not None:
                    # get minimum duration of track
                    min_duration = min(i.duration for i in infos)
                    # max_duration = max(i.duration for i in infos)
                    if min_duration > self.segment:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [musdb_license]
        return infos


musdb_license = dict()
