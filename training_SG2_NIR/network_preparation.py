
import dnnlib
from models.stylegan1 import Truncation
import torch
from collections import OrderedDict
from training import legacy
from torch_utils import misc

def prepare_SG2(resolution, path_to_pretrained, avg_latent, max_layer, gpus, device):
    
    spec = dnnlib.EasyDict(dict(ref_gpus= gpus, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=8))
    print(spec)
    res = resolution
    spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
    spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
    spec.fmaps = 1 if res >= 512 else 0.5
    spec.lrate = 0.002 if res >= 1024 else 0.0025
    spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
    spec.ema = spec.mb * 10 / 32

    G_kwargs = dnnlib.EasyDict(class_name='training_SG2_NIR.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())

    D_kwargs = dnnlib.EasyDict(class_name='training_SG2_NIR.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = D_kwargs.channel_base = int(spec.fmaps * 32768)
    G_kwargs.synthesis_kwargs.channel_max = D_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = spec.map
    G_kwargs.synthesis_kwargs.num_fp16_res = D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    G_kwargs.synthesis_kwargs.conv_clamp = D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    #args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    loss_kwargs = dnnlib.EasyDict(class_name='training_SG2_NIR.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    training_set_label_dim = 0
    training_set_resolution = resolution
    training_set_num_channels = 3 
    common_kwargs = dict(c_dim=training_set_label_dim, img_resolution=training_set_resolution, img_channels=training_set_num_channels)

    print(G_kwargs)
    print(common_kwargs)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).requires_grad_(False).to(device)

    print(G)
    print(f'Resuming from "{path_to_pretrained}"')
    with dnnlib.util.open_url(path_to_pretrained) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [('G', G)]:
        misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    ##########################################

    g_all = torch.nn.Sequential(OrderedDict([
        ('g_mapping', G.mapping),
        ('truncation', Truncation(avg_latent, max_layer=max_layer, device=device, threshold=0.7)),
        ('g_synthesis', G.synthesis)
    ]))


    return g_all, G_opt_kwargs, loss_kwargs
