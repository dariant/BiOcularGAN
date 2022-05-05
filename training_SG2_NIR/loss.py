# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from .plot_gradient_flow import plot_grad_flow

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, D_NIR, augment_pipe=None, style_mixing_prob=0.9, 
                 r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, TRAIN_IMAGE_OR_NIR_OR_MASK = "image"):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.pl_mean_NIR = torch.zeros([], device=device)

        self.D_NIR = D_NIR

        self.TRAIN_IMAGE_OR_NIR_OR_MASK = TRAIN_IMAGE_OR_NIR_OR_MASK
        #self.D_seg = D_seg
    # Run the generator
    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, NIR_img = self.G_synthesis(ws)
        return img, NIR_img, ws

    # Run the discriminator
    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    # Run the discriminator
    def run_D_NIR(self, NIR_img, c, sync):
        if self.augment_pipe is not None:
            NIR_img = self.augment_pipe(NIR_img)
        with misc.ddp_sync(self.D_NIR, sync):
            logits_NIR = self.D_NIR(NIR_img, c)
        return logits_NIR

    # Run the mask discriminator
    def run_D_mask(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D_mask, sync):
            logits = self.D_mask(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_NIR, real_c, gen_z, gen_c, sync, gain):
        #print(phase)
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_NIRmain', 'D_NIRreg']#'D_maskmain', 'D_maskreg', 'D_maskboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        do_D_NIR_main = (phase in ['D_NIRmain', 'Dboth'])
        do_D_NIR_r1   = (phase in ['D_NIR_reg', 'Dboth']) and (self.r1_gamma != 0)
        
        #do_Dmask_main = (phase in ['D_maskmain', 'D_maskboth'])
        #do_D_mask_r1   = (phase in ['D_maskreg', 'D_maskboth']) and (self.r1_gamma != 0)

        if self.TRAIN_IMAGE_OR_NIR_OR_MASK == "image" and False:
            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                #print("GMAIN")
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    gen_img, gen_mask, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                    
                    if self.TRAIN_IMAGE_OR_NIR_OR_MASK == "image":
                        gen_logits = self.run_D(gen_img, gen_c, sync=False)
                        training_stats.report('Loss/scores/fake', gen_logits)
                        training_stats.report('Loss/signs/fake', gen_logits.sign())

                        #gen_mask_logits = self.run_D_mask(gen_mask, gen_c, sync=False)
                        # compute the loss with SoftPlus
                        loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                    
                        training_stats.report('Loss/G/loss', loss_Gmain)

                        with torch.autograd.profiler.record_function('Gmain_backward'):
                            #print("-- Backward")
                            loss_Gmain.mean().mul(gain).backward()
                            #plot_grad_flow(self.G_mapping.named_parameters())
                            #plot_grad_flow(self.G_synthesis.named_parameters())
                            
            # Gpl: Apply path length regularization.
            if do_Gpl:
                #print("GPL")
                with torch.autograd.profiler.record_function('Gpl_forward'):
                    batch_size = gen_z.shape[0] // self.pl_batch_shrink
                    gen_img, gen_mask, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                    
                    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    
                    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    
                    pl_penalty = (pl_lengths - pl_mean).square()
                    training_stats.report('Loss/pl_penalty', pl_penalty)
                    
                    loss_Gpl = pl_penalty * self.pl_weight
                    training_stats.report('Loss/G/reg', loss_Gpl)

                with torch.autograd.profiler.record_function('Gpl_backward'):
                    (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

            # Dmain: Minimize logits for generated images.
            # get loss for Discriminator and backward it
            loss_Dgen = 0
            if do_Dmain:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    gen_img, gen_mask, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                    gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                with torch.autograd.profiler.record_function('Dgen_backward'):
                    loss_Dgen.mean().mul(gain).backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            if do_Dmain or do_Dr1:
                name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    real_img_tmp = real_img.detach().requires_grad_(do_Dr1)

                    real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())

                    loss_Dreal = 0
                    if do_Dmain:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if do_Dr1:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        ######################################
        # Do loss for both IMG and NIR 
        if self.TRAIN_IMAGE_OR_NIR_OR_MASK == "NIR":
            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                #print("GMAIN")
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    gen_img, gen_NIR, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                    
                    gen_logits = self.run_D(gen_img, gen_c, sync=False)
                    gen_NIR_logits = self.run_D_NIR(gen_NIR, gen_c, sync=False)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    
                    # compute the loss with SoftPlus
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                    # TODO should we do it like this
                    loss_Gmain_NIR = torch.nn.functional.softplus(-gen_NIR_logits) 
                    loss_Gmain = loss_Gmain + loss_Gmain_NIR
                    training_stats.report('Loss/G/loss', loss_Gmain)

                    with torch.autograd.profiler.record_function('Gmain_backward'):
                        #print("-- Backward")
                        loss_Gmain.mean().mul(gain).backward()
                        #plot_grad_flow(self.G_mapping.named_parameters())
                        #plot_grad_flow(self.G_synthesis.named_parameters())
                    

            # Gpl: Apply path length regularization.
            # TODO add to this? 
            if do_Gpl:
                #print("GPL")
                with torch.autograd.profiler.record_function('Gpl_forward'):
                    batch_size = gen_z.shape[0] // self.pl_batch_shrink
                    gen_img, gen_NIR, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                    
                    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    pl_noise_NIR = torch.randn_like(gen_NIR) / np.sqrt(gen_NIR.shape[2] * gen_NIR.shape[3])
                    
                    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                        pl_grads = pl_grads + torch.autograd.grad(outputs=[(gen_NIR * pl_noise_NIR).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    #with torch.autograd.profiler.record_function('pl_grads_NIR'), conv2d_gradfix.no_weight_gradients():
                    #    pl_grads_NIR = torch.autograd.grad(outputs=[(gen_NIR * pl_noise_NIR).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]

                    
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    
                    # TODO add this part
                    #pl_lengths_NIR = pl_grads_NIR.square().sum(2).mean(1).sqrt()
                    #pl_mean_NIR = self.pl_mean_NIR.lerp(pl_lengths_NIR.mean(), self.pl_decay)
                    #self.pl_mean_NIR.copy_(pl_mean_NIR.detach())
                    


                    pl_penalty = (pl_lengths - pl_mean).square()
                    training_stats.report('Loss/pl_penalty', pl_penalty)
                    
                    loss_Gpl = pl_penalty * self.pl_weight
                    training_stats.report('Loss/G/reg', loss_Gpl)

                with torch.autograd.profiler.record_function('Gpl_backward'):
                    (gen_img[:, 0, 0, 0] * 0 + gen_NIR[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

            # Dmain: Minimize logits for generated images.
            # get loss for Discriminator and backward it
            loss_Dgen = 0

            gen_img, gen_NIR, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            loss_Dgen_NIR = 0
            if do_Dmain:
                #print("Dmain")
                #print(self.D)#.requires_grad)
                #print(self.D_NIR.requires_grad)
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    #print("AAA")
                    gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.

                    #gen_logits_NIR = self.run_D_NIR(gen_NIR, gen_c, sync=False) # Gets synced by loss_Dreal.
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                    #loss_Dgen_NIR =  torch.nn.functional.softplus(gen_logits_NIR)

                    #print(loss_Dgen)
                    #print(loss_Dgen_NIR)

                    #print("end forward ")
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    #print("Dgen backward")
                    loss_Dgen.mean().mul(gain).backward()
                    #loss_Dgen_NIR.mean().mul(gain).backward()

            # Get loss of NIR discriminator
            
            if do_D_NIR_main:
                #print("D_NIR_main")
                #print(self.D)#.requires_grad)
                #print(self.D_NIR.requires_grad)
                with torch.autograd.profiler.record_function('Dgen_forward_NIR'):
                    #print("BBB")
                    # gen_img, gen_NIR, _ = self.run_G(gen_z, gen_c, sync=False)
                    gen_logits_NIR = self.run_D_NIR(gen_NIR, gen_c, sync=False) # Gets synced by loss_Dreal.

                    training_stats.report('Loss/scores/fake', gen_logits_NIR)
                    training_stats.report('Loss/signs/fake', gen_logits_NIR.sign())
                    loss_Dgen_NIR = torch.nn.functional.softplus(gen_logits_NIR) # -log(1 - sigmoid(gen_logits))
                    #print(loss_Dgen_NIR)
                    #print(loss_Dgen_NIR)

                    #print("end forward ")
                with torch.autograd.profiler.record_function('Dgen_backward_NIR'):
                    #print("Dgen NIR backward")
                    loss_Dgen_NIR.mean().mul(gain).backward()
                    #loss_Dgen_NIR.mean().mul(gain).backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            # TODO maybe problems in here? most likely??
            if do_Dmain or do_Dr1:
                #print("Dmain, Dr1")
                name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                    real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())
                    
                    #real_NIR_tmp = real_NIR.detach().requires_grad_(do_D_NIR_r1)
                    #real_logits_NIR = self.run_D_NIR(real_NIR_tmp, real_c, sync=sync)

                    loss_Dreal = 0
                    if do_Dmain:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                        #loss_Dreal = loss_Dreal + torch.nn.functional.softplus(-real_logits_NIR)
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if do_Dr1:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                            #r1_grads = r1_grads + torch.autograd.grad(outputs=[real_logits_NIR.sum()], inputs=[real_NIR_tmp], create_graph=True, only_inputs=True)[0]
                            # TODO this??? 
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            


            ########
            if do_D_NIR_main or do_D_NIR_r1:
                #print("Dmain, Dr1")
                name = 'Dreal_Dr1_NIR' if do_D_NIR_main and do_D_NIR_r1 else 'Dreal_NIR' if do_D_NIR_main else 'Dr1_NIR'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    real_NIR_tmp = real_NIR.detach().requires_grad_(do_D_NIR_r1)
                    real_logits_NIR = self.run_D_NIR(real_NIR_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real_NIR', real_logits_NIR)
                    training_stats.report('Loss/signs/real_NIR', real_logits_NIR.sign())
                    
                    #real_NIR_tmp = real_NIR.detach().requires_grad_(do_D_NIR_r1)
                    #real_logits_NIR = self.run_D_NIR(real_NIR_tmp, real_c, sync=sync)

                    loss_Dreal_NIR = 0
                    if do_D_NIR_main:
                        loss_Dreal_NIR = torch.nn.functional.softplus(-real_logits_NIR) # -log(sigmoid(real_logits))
                        #loss_Dreal = loss_Dreal + torch.nn.functional.softplus(-real_logits_NIR)
                        training_stats.report('Loss/D/loss_NIR', loss_Dgen_NIR + loss_Dreal_NIR)

                    loss_Dr1_NIR = 0
                    if do_D_NIR_r1:
                        with torch.autograd.profiler.record_function('r1_grads_NIR'), conv2d_gradfix.no_weight_gradients():
                            r1_grads_NIR = torch.autograd.grad(outputs=[real_logits_NIR.sum()], inputs=[real_NIR_tmp], create_graph=True, only_inputs=True)[0]
                            #r1_grads = r1_grads + torch.autograd.grad(outputs=[real_logits_NIR.sum()], inputs=[real_NIR_tmp], create_graph=True, only_inputs=True)[0]
                            # TODO this??? 
                        r1_penalty_NIR = r1_grads_NIR.square().sum([1,2,3])
                        loss_Dr1_NIR = r1_penalty_NIR * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty_NIR', r1_penalty_NIR)
                        training_stats.report('Loss/D/reg_NIR', loss_Dr1_NIR)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (real_logits_NIR * 0 + loss_Dreal_NIR + loss_Dr1_NIR).mean().mul(gain).backward()

        ######################################
        # TODO add loss for masks ... only activates when we set it into this training mode 
        elif self.TRAIN_IMAGE_OR_NIR_OR_MASK == "mask" and False:
            #print("TODO") # TODO 

            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                #print("MASK_GMAIN")
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    gen_img, gen_mask, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                    
                    if self.TRAIN_IMAGE_OR_NIR_OR_MASK == "image":
                        #gen_logits = self.run_D(gen_img, gen_c, sync=False)
                        #training_stats.report('Loss/scores/fake', gen_logits)
                        #training_stats.report('Loss/signs/fake', gen_logits.sign())

                        gen_mask_logits = self.run_D_mask(gen_mask, gen_c, sync=False)
                        # compute the loss with SoftPlus
                        loss_Gmain = torch.nn.functional.softplus(-gen_mask_logits) # -log(sigmoid(gen_logits))
                        training_stats.report('Loss/scores/fake_masks', gen_mask_logits)
                        training_stats.report('Loss/signs/fake_masks', gen_mask_logits.sign())
                        training_stats.report('Loss/G/loss_mask', loss_Gmain)

                        with torch.autograd.profiler.record_function('Gmain_masks_backward'):
                            print("-- Backward")
                            loss_Gmain.mean().mul(gain).backward()
                            #plot_grad_flow(self.G_mapping.named_parameters())
                            #plot_grad_flow(self.G_synthesis.named_parameters())
                            
            # Gpl: Apply path length regularization.


            """ if do_Gpl: # TODO 
                print("GPL")
                with torch.autograd.profiler.record_function('Gpl_forward'):
                    batch_size = gen_z.shape[0] // self.pl_batch_shrink
                    gen_img, gen_mask, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                    
                    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    
                    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    
                    pl_penalty = (pl_lengths - pl_mean).square()
                    training_stats.report('Loss/pl_penalty', pl_penalty)
                    
                    loss_Gpl = pl_penalty * self.pl_weight
                    training_stats.report('Loss/G/reg', loss_Gpl)

                with torch.autograd.profiler.record_function('Gpl_backward'):
                    (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward() """

            # Dmain: Minimize logits for generated images.
            # get loss for Discriminator and backward it
            loss_Dgen = 0
            if do_Dmask_main:
                #print("D_MASK")
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    gen_img, gen_mask, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                    gen_logits = self.run_D_mask(gen_mask, gen_c, sync=False) # Gets synced by loss_Dreal.
                    training_stats.report('Loss/scores/fake_masks', gen_logits)
                    training_stats.report('Loss/signs/fake_masks', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                with torch.autograd.profiler.record_function('Dgen_masks_backward'):
                    loss_Dgen.mean().mul(gain).backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            if do_Dmask_main or do_D_mask_r1:
                #print("D_mask_REG")
                name = 'Dreal_Dr1_masks' if do_Dmask_main and do_D_mask_r1 else 'Dreal_masks' if do_Dmask_main else 'D_masks_r1'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    real_mask_tmp = real_mask.detach().requires_grad_(do_D_mask_r1)

                    real_logits = self.run_D_mask(real_mask_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real_masks', real_logits)
                    training_stats.report('Loss/signs/real_masks', real_logits.sign())

                    loss_Dreal = 0
                    if do_Dmask_main:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D_masks/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if do_D_mask_r1:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_mask_tmp], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty_masks', r1_penalty)
                        training_stats.report('Loss/D_masks/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            # idea: 
            # lock the entirety of the model ... except for the mask layers 
            # make a loss for those ? train them? 
#----------------------------------------------------------------------------
