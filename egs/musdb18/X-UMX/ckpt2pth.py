import os, sys
import argparse
import json
import yaml

import torch
import pytorch_lightning as pl
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger
import asteroid
from asteroid import engine
import numpy as np
from asteroid.models import XUMX
import datasets
import models
# import losses
import utils



pl.seed_everything(1, workers=True)

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "--checkpoint", type=str, required=True, help="the checkpoint to export as best model"
)

def main(conf,args):

    # dataloader_kwargs = ({})

    # train_set = datasets.CombineDatasets(
    #     speech_dirs=conf["data"]["speech_train_dir"],
    #     music_dirs=conf["data"]["music_train_dir"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     original_sample_rate=conf["data"]["original_sample_rate"],
    #     segment=conf["data"]["segment"],
    #     shuffle_tracks=True,
    #     multi_speakers=conf["training"]["multi_speakers"],
    #     multi_speakers_frequency=conf["training"]["multi_speakers_frequency"],
    #     data_ratio=conf["training"]["data_ratio"] if "data_ratio" in conf["training"] else 1.,
    #     new_data=conf["training"]["new_data"] if "new_data" in conf["training"] else False
    # )
    # val_set = datasets.CombineDatasets(
    #     speech_dirs=conf["data"]["speech_valid_dir"],
    #     music_dirs=conf["data"]["music_valid_dir"],
    #     sample_rate=conf["data"]["sample_rate"],
    #     original_sample_rate=conf["data"]["original_sample_rate"],
    #     segment=conf["data"]["segment"],
    #     shuffle_tracks=False,
    #     multi_speakers=conf["training"]["multi_speakers"],
    #     multi_speakers_frequency=conf["training"]["multi_speakers_frequency"],
    #     data_ratio=1.,
    #     new_data=False
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     shuffle=True,
    #     batch_size=conf["training"]["batch_size"],
    #     drop_last=True,
    #     **dataloader_kwargs
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     shuffle=False,
    #     batch_size=conf["training"]["batch_size"],
    #     drop_last=True,
    #     pin_memory=torch.cuda.is_available(),
    #     num_workers=conf["training"]["num_workers"],
    # )

    ###Models
    model = XUMX(
        window_length=args.window_length,
        input_mean=None,
        input_scale=None,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        in_chan=args.in_chan,
        n_hop=args.nhop,
        sources=args.sources,
        max_bin=torch.NoneType,
        bidirectional=args.bidirectional,
        sample_rate=args.sample_rate,
        spec_power=args.spec_power,
        return_time_signals=True if args.loss_use_multidomain else False,
    )
    # if(conf["model"]["name"] == "ConvTasNet"):
    #     conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    #     model = models.ConvTasNetNorm(
    #         conf["filterbank"],
    #         conf["masknet"],
    #         sample_rate=conf["data"]["sample_rate"],
    #         device=arg_dic["main_args"]["device"]
    #     )
    # elif (conf["model"]["name"] == "UNet"):
    #     # UNet with logl2 time loss and normalization inside model
    #     model = models.UNet(
    #         conf["data"]["sample_rate"],
    #         conf["data"]["fft_size"],
    #         conf["data"]["hop_size"],
    #         conf["data"]["window_size"],
    #         conf["model"]["kernel_size"],
    #         conf["model"]["stride"],
    #         device=arg_dic["main_args"]["device"],
    #         mask_logit=conf["model"]["mask_logit"]
    #     )
    # elif (conf["model"]["name"] == "UNetP"):
    #     # UNet with logl2 time loss and normalization inside model
    #     model = models.UNetP(
    #         conf["data"]["sample_rate"],
    #         conf["data"]["fft_size"],
    #         conf["data"]["hop_size"],
    #         conf["data"]["window_size"],
    #         conf["model"]["kernel_size"],
    #         conf["model"]["stride"],
    #         device=arg_dic["main_args"]["device"],
    #         mask_logit=conf["model"]["mask_logit"]
    #     )
    # elif(conf["model"]["name"] == "OUMX"):
    #     scaler_mean, scaler_std = datasets.get_statistics(conf, val_set)
    #     max_bin = utils.bandwidth_to_max_bin(conf["data"]["sample_rate"], conf["model"]["in_chan"], conf["model"]["bandwidth"])
    #     #scaler_mean, scaler_std = np.array([0]*max_bin), np.array([0.5]*max_bin)
    #     model = models.OUMX(
    #         window_length=conf["model"]["window_length"],
    #         input_mean=scaler_mean,
    #         input_scale=scaler_std,
    #         nb_channels=conf["model"]["nb_channels"],
    #         hidden_size=conf["model"]["hidden_size"],
    #         in_chan=conf["model"]["in_chan"],
    #         n_hop=conf["model"]["nhop"],
    #         sources=conf["data"]["sources"],
    #         max_bin=max_bin,
    #         bidirectional=conf["model"]["bidirectional"],
    #         sample_rate=conf["data"]["sample_rate"],
    #         spec_power=conf["model"]["spec_power"],
    #     )
    # elif(conf["model"]["name"] == "UNetAttn"):
    #     #scaler_mean, scaler_std = datasets.get_statistics(conf, val_set)
    #     max_bins = utils.bandwidth_to_max_bin(conf["data"]["sample_rate"], conf["model"]["in_chan"], conf["model"]["bandwidth"])
    #     #scaler_mean, scaler_std = np.array([0]*max_bin), np.array([0.5]*max_bin)
    #     model = models.UNetAttn(
    #         window_length=conf["model"]["window_length"],
    #         nb_channels=conf["model"]["nb_channels"],
    #         in_chan=conf["model"]["in_chan"],
    #         n_hop=conf["model"]["nhop"],
    #         max_bins=max_bins,
    #         sample_rate=conf["data"]["sample_rate"],
    #         spec_power=conf["model"]["spec_power"],
    #         n_src=conf["data"]["n_src"],
    #         mask_logit=conf["model"]["mask_logit"],
    #         k=conf["model"]["k"],
    #         hidden_size=conf["model"]["hidden_size"]
    #     )

    # ###Losses
    # loss_module = utils.my_import("losses."+conf["training"]['loss'])
    # if 'weighted' in conf["training"]['loss']:
    #     loss_func = loss_module(weights=np.array(conf["training"]['class_weights'].split(';'),dtype=np.float32))
    # elif 'MultiDomain' in conf["training"]['loss']:
    #     loss_func = losses.MultiDomainLoss(
    #         window_length=conf["model"]["window_length"],
    #         in_chan=conf["model"]["in_chan"],
    #         n_hop=conf["model"]["nhop"],
    #         spec_power=conf["model"]["spec_power"],
    #         nb_channels=conf["model"]["nb_channels"],
    #         loss_combine_sources=conf["training"]['loss_combine_sources'],
    #         loss_use_multidomain=conf["training"]['loss_use_multidomain'],
    #         mix_coef=conf["training"]['mix_coef'],
    #     )
    # else:
    #     loss_func = loss_module()

    # ###Optimizer
    # optimizer = asteroid.engine.optimizers.make_optimizer(model.parameters(), lr=conf["optim"]["lr"], weight_decay=conf["optim"]["weight_decay"])
    # if conf["training"]["half_lr"]:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer=optimizer, factor=conf["optim"]["lr_decay_gamma"], patience=conf["optim"]["lr_decay_patience"], cooldown=10
    #     )

    # # Just after instantiating, save the args. Easy loading in the future.
    # exp_dir = os.path.join(conf["main_args"]["exp_dir"],conf["main_args"]['config'].split('.yml')[0])
    # os.makedirs(exp_dir, exist_ok=True)
    # conf_path = os.path.join(exp_dir, "conf.yml")
    # with open(conf_path, "w") as outfile:
    #     yaml.safe_dump(conf, outfile)

    # ### delete paths key which raises error
    # if "speech_train_dir" in conf["data"]: del conf["data"]["speech_train_dir"]
    # if "speech_valid_dir" in conf["data"]: del conf["data"]["speech_valid_dir"]
    # if "music_train_dir" in conf["data"]: del conf["data"]["music_train_dir"]
    # if "music_valid_dir" in conf["data"]: del conf["data"]["music_valid_dir"]

    # system = asteroid.engine.system.System(
    #     model=model,
    #     loss_func=loss_func,
    #     optimizer=optimizer,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     scheduler=scheduler,
    #     config=conf
    # )

    # callbacks = []
    # checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    # checkpoint = pl.callbacks.ModelCheckpoint(
    #     checkpoint_dir,
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=5,
    #     verbose=True
    # )
    # callbacks.append(checkpoint)
    # if conf["training"]["early_stop"]:
    #     callbacks.append(pl.callbacks.EarlyStopping(
    #         monitor="val_loss",
    #         mode="min",
    #         patience=conf["optim"]["patience"],
    #         verbose=True
    #     ))

    # if conf["main_args"]["load"]:
    #     checkpoint_files = [os.path.join(checkpoint_dir, x) for x in os.listdir(checkpoint_dir) if x.endswith(".ckpt")]
    #     if len(checkpoint_files)>0:
    #         newest_checkpoint = max(checkpoint_files, key = os.path.getctime)
    #     else:
    #         newest_checkpoint = None
    # else:
    #     newest_checkpoint = None


    device = torch.device(conf["main_args"]["device"])
    state_dict = torch.load(conf["main_args"]["checkpoint"], map_location=device)

    to_save = model.serialize()
    #to_save.update(train_set.get_infos())
    torch.save(to_save, conf["main_args"]["output_model"])


if __name__ == "__main__":
    args = parser.parse_args()

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.

    config_model = os.path.join(os.path.dirname(args.checkpoint).replace(os.path.sep+'checkpoints',''), "conf.yml")
    with open(config_model) as f:
        def_conf = yaml.safe_load(f)
    parser = asteroid.utils.prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = asteroid.utils.parse_args_as_dict(parser, return_plain_args=True)
    arg_dic["main_args"]["device"] = 'cpu'
    arg_dic["main_args"]["output_model"] = os.path.join(os.path.dirname(args.checkpoint).replace(os.path.sep+'checkpoints',''), "best_model.pth")
    main(arg_dic,plain_args)

