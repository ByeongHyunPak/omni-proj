def get_erp_up_noise_pred(noise_preds, pers_indices):
    B, C, H_tgt, W_tgt = noise_preds[0].shape
    H_up_src, W_up_src = pers_indices.shape

    device = noise_preds[0].device

    # Initialize result tensors
    erp_up_noise_pred_flat = torch.zeros(B, C, 1+H_up_src*W_up_src, device=device)
    erp_up_noise_counts_flat = torch.zeros(B, C, 1+H_up_src*W_up_src, device=device)

    # Loop over residual_pers_noises and indices
    for noise_pred, indices in zip(noise_preds, pers_indices):
        pass



class ERPMultiDiffusion_v3_3(MultiDiffusion):
    
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__(device, sd_version, hf_key)
        self.up_level = 3
        self.tgt_cfg = {
            # "view_dirs": [
            #     (0,0), (0,30), (0,-30), (0,60), (0,-60), (0,90), (0,-90), (0,120), (0,-120),
            #     (22.5, 0), (22.5, 30), (22.5, -30), (22.5, 60), (22.5, -60), (22.5, 90), (22.5, -90), (22.5, 90), (22.5, 120), (22.5, -120),
            #     (45.0, 0), (45.0, 30), (45.0, -30), (45.0, 60), (45.0, -60), (45.0, 90), (45.0, -90), (45.0, 90), (45.0, 120), (45.0, -120),
            # ]
            # "view_dirs": [
            #     (0.0, -45.0), (30.0, -45.0), (60.0, -45.0), (90.0, -45.0), (-30.0, -45.0), (-60.0, -45.0), (-90.0, -45.0),
            #     (0.0, -22.5), (30.0, -22.5), (60.0, -22.5), (90.0, -22.5), (-30.0, -22.5), (-60.0, -22.5), (-90.0, -22.5),
            #     (0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (90.0, 0.0), (-30.0, 0.0), (-60.0, 0.0), (-90.0, 0.0),
            #     (0.0, 22.5), (30.0, 22.5), (60.0, 22.5), (90.0, 22.5), (-30.0, 22.5), (-60.0, 22.5), (-90.0, 22.5),
            #     (0.0, 45.0), (30.0, 45.0), (60.0, 45.0), (90.0, 45.0), (-30.0, 45.0), (-60.0, 45.0), (-90.0, 45.0),
            # ]
            # "view_dirs": [
            #     (0.0, -22.5), (30.0, -22.5), (60.0, -22.5), (-30.0, -22.5), (-60.0, -22.5),
            #     (0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (-30.0, 0.0), (-60.0, 0.0),
            #     (0.0, 22.5), (30.0, 22.5), (60.0, 22.5), (-30.0, 22.5), (-60.0, 22.5),
            # ]
            "view_dirs": [
                (0.0, -22.5), (30.0, -22.5), (-30.0, -22.5),
                (0.0, 0.0), (30.0, 0.0), (-30.0, 0.0),
                (0.0, 22.5), (30.0, 22.5), (-30.0, 22.5),
            ]
        }
    
    @torch.no_grad()
    def text2erp(self,
                 prompts, 
                 negative_prompts='', 
                 height=512, width=1024, 
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 visualize_intermidiates=False,
                 save_dir=None):
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define ERP source noise
        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.no_grad():
            
            if visualize_intermidiates is True:
                intermidiate_imgs = []
                
            self.tgt_cfg["size"] = (64, 64)
            pers_latents, pers_indices, erp_up_noise, fin_v_num =\
                get_pers_view_noises(latent.to("cpu"), self.up_level, self.tgt_cfg)

            for i, t in enumerate(tqdm(self.scheduler.timesteps)):

                denoised_pers_latents = []
                noise_preds = []

                for latent_view in pers_latents:
                    
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                    denoised_pers_latents.append(latents_view_denoised)
                    
                    # compute residual noise
                    noise_pred = noise_pred / torch.sqrt(fin_v_num)
                    noise_preds.append(noise_pred)

                erp_up_noise_pred = get_erp_up_noise_pred(noise_preds, pers_indices)

                erp_up_noise_denoised = self.scheduler.step(erp_up_noise_pred, t, erp_up_noise)

                pers_latents, _, erp_up_noise, _ =\
                    get_pers_view_noises(erp_up_noise_denoised, 1, self.tgt_cfg)

                # visualize intermidiate timesteps
                if visualize_intermidiates is True:
                    pers_img_inps = []
                    for k, pers_latent in enumerate(pers_latents):
                        pers_img = self.decode_latents(pers_latent)
                        pers_img_inps.append((self.tgt_cfg['view_dirs'][k], pers_img))
                    intermidiate_imgs.append((i+1, pers_img_inps))
                
                if save_dir is not None:
                    # save image
                    if os.path.exists(f"{save_dir}/{i:0>2}") is False:
                        os.mkdir(f"{save_dir}/{i:0>2}/")
                    for v, im in pers_img_inps:
                        theta, phi = v
                        im = ToPILImage()(im[0].cpu())
                        im.save(f'/{save_dir}/{i:0>2}/pers_{theta}_{phi}.png')
        
        return intermidiate_imgs
