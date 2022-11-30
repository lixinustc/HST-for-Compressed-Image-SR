# HST
HST: Hierarchical Swin Transformer for Compressed Image Super-resolution
> [**HST**](https://arxiv.org/abs/2208.09885), Bingchen Li, Xin Li, et al.    

> Achieved **the fifth place** in the competition of the **AIM2022 compressed image super-resolution** track.

> Accepted by ECCV2022 Workshop 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-div2k)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-div2k?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-div2k-1)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-div2k-1?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-div2k-2)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-div2k-2?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-div2k-3)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-div2k-3?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-set5-q10)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-set5-q10?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-set14)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-set14?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-bsd100)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-bsd100?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-urban100)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-urban100?p=hst-hierarchical-swin-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hst-hierarchical-swin-transformer-for/compressed-image-super-resolution-on-manga109)](https://paperswithcode.com/sota/compressed-image-super-resolution-on-manga109?p=hst-hierarchical-swin-transformer-for)

![image](https://github.com/lixinustc/HST-Hierarchical-Swin-Transformer-for-Compressed-Image-Super-resolution/blob/main/figs/HST.png)

## Abstract 
Compressed Image Super-resolution has achieved great attention in recent years, where images are degraded with compression artifacts and low-resolution artifacts. Since the complex hybrid distortions,
it is hard to restore the distorted image with the simple cooperation
of super-resolution and compression artifacts removing. In this paper,
we take a step forward to propose the Hierarchical Swin Transformer
(HST) network to restore the low-resolution compressed image, which
jointly captures the hierarchical feature representations and enhances
each-scale representation with Swin transformer, respectively. Moreover,
we find that the pretraining with Super-resolution (SR) task is vital
in compressed image super-resolution. To explore the effects of different SR pretraining, we take the commonly-used SR tasks (e.g., bicubic
and different real super-resolution simulations) as our pretraining tasks,
and reveal that SR plays an irreplaceable role in the compressed image super-resolution. With the cooperation of HST and pre-training, our
HST achieves the fifth place in AIM 2022 challenge on the low-quality
compressed image super-resolution track, with the PSNR of 23.51dB. Extensive experiments and ablation studies have validated the effectiveness
of our proposed methods.

## Usages
More details will be decribed progressively.

**The checkpoints for HST are released**:
- [checkpoint_comp10_x4](https://drive.google.com/file/d/1ZtGxO6ghT1YFLgu_PIHBt7VpDV52CsjS/view?usp=sharing)
- [checkpoint_comp20_x4](https://drive.google.com/file/d/1ldXbI5c9KHxsHQZS3hRRK2jR9HRvfqyD/view?usp=sharing)
- [checkpoint_comp30_x4](https://drive.google.com/file/d/1ANqQkYW7JKPLdJLKq1xHixaSZtn3e0q-/view?usp=sharing)
- [checkpoint_comp40_x4](https://drive.google.com/file/d/1SlvhcFSEr4jM5gUB_we8c-EYmpIUntIT/view?usp=sharing)


## Cite US
Please cite us if this work is helpful to you.
```
@inproceedings{li2022hst, 
title={HST: Hierarchical Swin Transformer for Compressed Image Super-resolution}, 
   author={Li, Bingchen and Li, Xin and Lu, Yiting and Liu, Sen and Feng, Ruoyu and Chen, Zhibo}, 
   booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshops}, 
   year={2022} 
}
```

The model is implemented based on the works: 
[MSGDN](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w7/Li_Multi-Scale_Grouped_Dense_Network_for_VVC_Intra_Coding_CVPRW_2020_paper.pdf), [SwinIR](https://github.com/JingyunLiang/SwinIR), [SwinTransformer](https://arxiv.org/abs/2103.14030)
