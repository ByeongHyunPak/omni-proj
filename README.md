# omni-proj
Omnidirectional image projection

## A. Goal

- High-Resolution 360-degree Panorama Image Generation

## B. Main Idea
<details>
<summary> 1. ERP-MultiDiffusion (v1) + Sphere-INFD (24.10.21 ~ 27) - [Sec. C1]</summary>

- Step 1) ERP-MultiDiffusion (v1): LR ERP Image Generation w/ MultiDiffusion
         
    - MultiDiffusion = pre-trained Stable Diffusion으로 512x512 해상도 이상의 panorama image 생성 (zero-shot) [1]
    
    - (idea) 각 perspective image crop을 병렬적으로 denoising 하면서, 각 denoising step마다 ERP plane에 projection하고 fuse.
         
        - = perspective patch들 사이 overlap region의 consistency + ERP geometry
     
- Step 2) INFD + Spherical Coordinate = Upsample 360-degree Panorama Image

    - Image Neural Field Diffusion (INFD) models = Image neural field 생성 -> continuous image representation [2]
    
    - (idea) INFD + Spherical Coordinate = continuous 360-degree panorama image representation (like [3])

</details>

<details>
<summary> 2. ERP-MultiDiffusion (v2) + Sphere-INFD (24.10.30 ~ ) - [Sec. C2] &rarr; Progressing... </summary>

- ERP-MultiDiffusion (v1): denoising step 마다 perspective patch &rarr; proj. & fuse on ERP plane &rarr; perspective patch

    - ERP &lrarr; Pers. projection이 각 perspective patch의 initial noise의 분포를 normal distribution이 아닌 왜곡된 분포를 따르게 함.
     
    - 그 결과, 매우 망가지는 이미지 생성 (Sec. C1)
 
- ERP-MultiDiffusion (v2): ERP-MultiDiffusion w/o ERP &lrarr; Pers. projection

    - MultiDiffusion의 regular gird sliding window denoising은 그대로 유지
     
    - `prompt=f"360-degree panoramic image, {prompt}"`
   
    - `circular_padding` : left/rightmost의 continuity를 위해서

    - (TBD)

</details>

## C. Experiment Results

<details>
<summary> 1. ERP-MultiDiffusion (v1) zero-shot results (24.10.21 ~ 27) - [Sec. B1]  </summary>

- "Firenze Cityscape"
![Image](https://github.com/user-attachments/assets/7a4c3315-a5fe-4298-aa40-5abe67fa1869)

- "Japanese anime style downtown city street"
![Image](https://github.com/user-attachments/assets/ad8fb8ee-cf0e-4274-a4c1-9bd3c7a10b89)

- More Details and Results: [link](https://drive.google.com/file/d/1421z-XUghglSKX3_0adQW70oxFp5wcQv/view?usp=sharing)

</details>

<details>
<summary>2. ERP-MultiDiffusion (v2) zero-shot results (24.10.30 ~ ) - [Sec. B2] &rarr; Progressing... </summary>

</details>

## References
[1] O. Bar-Tal et al. "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation", ICML 2023 ([arXiv](https://arxiv.org/pdf/2302.08113))
[2] Y. Chen et al. "Image Neural Field Diffusion Models", CVPR 2024 ([arXiv](https://arxiv.org/pdf/2406.07480))
[3] Y. Yoon et al. "SphereSR: 360-degree Image Super-Resolution with Arbitrary Projection via Continuous Spherical Image Representation", CVPR 2022 ([arXiv](https://arxiv.org/pdf/2112.06536))

## Related Works

<details>
<summary> Text &rarr; 360º Panorama Image </summary>

- [x] A. Liu et al. "PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance", ECCV 2024 ([arXiv](https://arxiv.org/pdf/2408.02157))

- [x] A. Voynov et al. "Curved Diffusion: A Generative Model With Optical Geometry Control", ECCV 2024 ([arXiv](https://arxiv.org/pdf/2311.17609))

- [x]  T. Wu et al. "SphereDiffusion: Spherical Geometry-Aware Distortion Resilient Diffusion Model", AAAI 2024 ([arXiv](https://arxiv.org/pdf/2403.10044))

- [x] H. Wang et al. "Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models", WACV 2024 ([arXiv](https://arxiv.org/pdf/2310.18840))

- [x] C. Zhang et al. "Taming Stable Diffusion for Text to 360º Panorama Image Generation", CVPR 2024 Highlight ([arXiv](https://arxiv.org/pdf/2404.07949))

- [ ] J. Li & M. Bansal "PANOGEN: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation", NeurIPS 2023 ([arXiv](https://arxiv.org/pdf/2305.19195))

- [ ] Q. Zhang et al. "DiffCollage: Parallel Generation of Large Content with Diffusion Models", CVPR 2023 ([arXiv](https://arxiv.org/pdf/2303.17076))

- [X] Z. Chen et al. "Text2Light: Zero-Shot Text-Driven HDR Panorama Generation", TOG 2022 ([arXiv](https://arxiv.org/pdf/2209.09898))

</details>


<details>
<summary> Text &rarr; 360º Panorama Video </summary>

- [x] Q. Wang et al. "360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model", CVPR 2024 ([arXiv](https://arxiv.org/pdf/2401.06578))

</details>


<details>
<summary> nFOV Image &rarr; 360º Panorama Image </summary>

- [x] A. Nakata & T. Yamanaka "2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction", ECCV 2024 ([arXiv](https://www.arxiv.org/pdf/2409.09969))

- [ ] J. Wang et al. "360-Degree Panorama Generation from Few Unregistered NFoV Images", ACM MM 2023 ([arXiv](https://arxiv.org/pdf/2308.14686))

- [ ] T. Hara et al. "Spherical Image Generation from a Single Normal Field of View Image by Considering Scene Symmetry", AAAI 2021 ([arXiv](https://arxiv.org/pdf/2001.02993))

</details>



<details>
<summary> Panorama-to-Panorama Translation </summary>

- [ ] H. Wang et al. "360PanT: Training-Free Text-Driven 360-Degree Panorama-to-Panorama Translation", WACV 2025 ([arXiv](https://arxiv.org/pdf/2409.08397))

</details>
