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
import numpy as np
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
        sources=["bass", "percussion", "vocal", "other"], 
        # the other should be the same in conf.yml
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
    ):

        self.root = Path(root).expanduser()
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
        #print(self.tracks)
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        # self.__getitem__(index = 1)

#     def __getitem__(self, index):
#         # create a dict for storing stem grouping rule
        
        
        
#         # assemble the mixture of target and interferers
#         audio_sources = {}

#         # get track_id
#         track_id = index // self.samples_per_track
        
        
#         # print("min duration = ", self.tracks[track_id]["min_duration"])
#         if self.random_segments:
#             start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
#         else:
#             start = 0

#         # create sources based on multitracks
#         # for source in self.sources:
#         # optionally select a random track for each source
#         if self.random_track_mix:
#             # load a different track
#             track_id = random.choice(range(len(self.tracks)))
#             if self.random_segments:
#                 start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)

#         # loads the full track duration
#         start_sample = int(start * self.sample_rate)
#         # check if dur is none
#         if self.segment:
#             # stop in soundfile is calc in samples, not seconds
#             stop_sample = start_sample + int(self.segment * self.sample_rate)
#         else:
#             # set to None for reading complete file
#             stop_sample = None

#         # load actual audio
# #             audio, _ = sf.read(
# #                 Path(self.tracks[track_id]["path"] / source).with_suffix(self.suffix),
# #                 always_2d=True,
# #                 start=start_sample,
# #                 stop=stop_sample,
# #             )
        
#         track_path = self.tracks[track_id]['path']
#         print("Track_Title = ",track_path)
#         # load multitracks and be ready to do linear mix
#         max_len = []
#         for i in self.grouping_info:
#             # print(i) # get source names
#             stem_tracks = []
#             # get all instrument name within one stem
#             for j in self.grouping_info[i]:
#                 # print(j)
#                 # get all multitrack filename within one instrument
#                 # get the corresponding song:
#                 track_title = str(track_path).split('\\')[-1]
#                 track_title = track_title.replace('_',' ')
                
#                 track_df = self.csv_info.loc[self.csv_info['Music_Title']==track_title]
#                 # print(track_df)
#                 if track_df.empty==True:
#                     print("This song doesn't exist in csv file", track_title)
                    
                    
                    
#                 temp = track_df[j].tolist()[0]
#                 if temp != '[]':
#                     for m in temp.strip('[]').split(', '):
#                         # print(m.strip(''))
#                         stem_tracks.append(m.strip('\''))
                
#             # apply linear mix within one source (stem) later can intergrate with data augmentation
#             # first load each multitrack
#             source_multitrack = {}
#             max_len_source = 0
#             for k in stem_tracks:
#                 audio,_ = sf.read(
#                     Path(self.tracks[track_id]['path'] / k),
#                 always_2d=True,
#                 start=start_sample,
#                 stop=stop_sample,
#                 )
#                 # convert to torch tensor
#                 audio = torch.tensor(audio.T, dtype=torch.float)
                
#                 if list(audio.shape)[0] == 1:
#                     audio = audio.repeat(2,1)
                    
#                 # apply multitrack-wise augmentations
#                 # audio = self.multitrack_augmentation(audio)
#                 if max_len_source <= audio.size(dim=1):
#                     max_len_source = audio.size(dim=1)
#                 source_multitrack[k] = audio
                
#             # zero-padding to the maximum length of tensor
#             max_len.append(max_len_source)

#             # print("Max_length = ",max_len_source)
#             for key, value in source_multitrack.items():
#                 if value.size(dim=1)==max_len_source:
#                     continue
#                 else:
#                     # print("before zero-padding, value.size = ",value.size(dim=1))
#                     target = torch.zeros(2,max_len_source)
#                     source_len = value.size(dim=1)
#                     target[:,:source_len] = value
#                     source_multitrack[key] = target
#                     # print("after padding, len = ", source_multitrack[key].size(dim=1))


            

#             # apply linear mix over all multitracks within one source index=0
#             source_mix = torch.stack(list(source_multitrack.values())).sum(0)
#             audio_sources[i] = source_mix
#             # apply source-wise augmentations
#             # source_mix = self.source_augmentations(source_mix)


        
#         for key, value in audio_sources.items():
#             if value.size(dim=1)==max(max_len):
#                 continue
#             else:
#                 # print("before zero-padding, value.size = ",value.size(dim=1))
#                 target = torch.zeros(2,max(max_len))
#                 source_len = value.size(dim=1)
#                 target[:,:source_len] = value
#                 audio_sources[key] = target
#                 # print("after padding, len = ", source_multitrack[key].size(dim=1))

#         audio_mix = torch.stack(list(audio_sources.values())).sum(0)
#         if self.targets:
#             audio_sources = torch.stack(
#                 [wav for src, wav in audio_sources.items() if src in self.targets], dim=0
#         )
#         # audio_mix a mixture over the sources, audio_sources is a concatenation of all sources
#         return audio_mix, audio_sources
    
    def __getitem__(self, index):
        # assemble the mixture of target and interferers
        audio_sources = {}

        # get track_id
        track_id = index // self.samples_per_track
        if self.random_segments:
            start = random.uniform(0, self.tracks[track_id]["min_duration"] - self.segment)
        else:
            start = 0

        # load sources
        for source in self.sources:
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
            track_path = self.tracks[track_id]["path"]

            if 'musdb' in self.root.name:
                mus_source = source
                if mus_source == 'percussion':
                    mus_source = 'drums'
                elif mus_source == 'vocal':
                    mus_source = 'vocals'
                source_path = os.path.join(track_path / (mus_source +  self.suffix))
            else:
                source_path = os.path.join(track_path / (track_path.name+ '_STEMS') / 'MUSDB'/ (track_path.name+ '_STEM_MUSDB_'+ source+ self.suffix))
            audio, _ = sf.read(
                source_path,
                always_2d=True,
                start=start_sample,
                stop=stop_sample,
            )
            if audio.shape[1] == 2:
                # if we have mono, let's duplicate it
                # as the input of OpenUnmix is always stereo
                audio = audio.sum(axis=1) / 2
                audio = np.expand_dims(audio, axis=1)
            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)
            # apply source-wise augmentations
            audio = self.source_augmentations(audio)
            audio_sources[source] = audio

        # apply linear mix over source index=0
        audio_mix = torch.stack(list(audio_sources.values())).sum(0)
        if self.targets:
            audio_sources = torch.stack(
                [wav for src, wav in audio_sources.items() if src in self.targets], dim=0
            )
        return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        """load tracks that contain all the required sources tracks"""
        p = Path(self.root, self.split) # train and test folder
        # p = Path(self.root)
        
        for track_path in tqdm.tqdm(p.iterdir()):
            
            if track_path.is_dir():
                if 'musdb' in self.root.name:
                    musdb_sources = []
                    for s in self.sources:
                        if s == 'percussion':
                            s = 'drums'
                        elif s == 'vocal':
                            s = 'vocals'
                        musdb_sources.append(s)
                    source_paths = [track_path / ( s + self.suffix) for s in musdb_sources] # 固定命名
                    
                else:
                    n_src_dir = len(os.listdir(track_path / (track_path.name+ '_STEMS') / 'MUSDB'))
                    if n_src_dir != 4:
                        #skip this track
                        print(f"Exclude track due to lack of 4 standard musdb tracks, only {n_src_dir}, {track_path}")
                        continue
                    if self.subset and track_path.name not in self.subset:
                        # skip this track
                        continue
                    # print(track_path)
                    
                    source_paths = [track_path / (track_path.name+ '_STEMS') / 'MUSDB'/ (track_path.name+ '_STEM_MUSDB_'+ s + self.suffix) for s in self.sources] # 固定命名
                    
                # get metadata
                infos = list(map(sf.info, source_paths))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print(infos[:].samplerate, self.sample_rate)
                    print("Exclude track due to different sample rate ", track_path)
                    continue

                if self.segment is not None:
                    # get minimum duration of track
                    min_duration = min(i.duration for i in infos)
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
