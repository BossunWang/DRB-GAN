# DRB-GAN
Implement about DRB-GAN: A DynamicResBlock Generative Adversarial Network for Artistic Style Transfer:
https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_DRB-GAN_A_Dynamic_ResBlock_Generative_Adversarial_Network_for_Artistic_Style_ICCV_2021_paper.pdf

## Implement detail
### Reference
* MUNIT: https://github.com/NVlabs/MUNIT/tree/a82e222bc359892bd0f522d7a0f1573f3ec4a485
  * MsImageDis, Basic Blocks, AdaptiveInstanceNorm2d
* FUNIT: https://github.com/NVlabs/FUNIT
  * ClassModelEncoder, ContentEncoder
* UGATIT: https://github.com/znxlwm/UGATIT-pytorch
  * Class Activation Map(setting bias=False),  Gamma, Beta block, adaILN
* Dynamic Convolution: https://github.com/TArdelean/DynamicConvolution
* VGG19: https://github.com/gordicaleksa/pytorch-neural-style-transfer/blob/master/models/definitions/vgg_nets.py
  
### Trick
* Dynamic Convolution weight from gammma
  * https://github.com/BossunWang/DRB-GAN/blob/642beae6e9a1437b77d3aa70d39870a7b2e87cfd/model/DynamicConv.py#L117
* Classifier weight without gradient calculate with style_mixed_feature
  * https://github.com/BossunWang/DRB-GAN/blob/642beae6e9a1437b77d3aa70d39870a7b2e87cfd/model/StyleEncodNet.py#L80
* Multi-scale discriminator
  * https://github.com/BossunWang/DRB-GAN/blob/642beae6e9a1437b77d3aa70d39870a7b2e87cfd/model/DiscriminativeNet.py#L32

## To Do List
- [ ] Explore data analysis using v2 model: cluster of style features, style classifer predicted labels difference from true label
- [ ] Traning with fp16
- [ ] Observered How to converge
- [ ] Rolling Guidance Filter, Guided Filter, Gabor Filter
- [X] Evaluation Metric: LPIPS, ArtFID, SIFID
- [ ] AdaWCT, FastDifferentiableMatSqrt
- [ ] Weighted averaging strategy for collection style transfer
- [ ] Different data augmentation(sharpness) with assigned artist's style


## Evaluation
### AFHQ
* LPIPS: Reference-guided LPIPS values for each style with content: 0.5225579113960266
* FID between style and stylized: 49.61556248811253
* SIFID between style and stylized: 0.00016049967
### Photo Scene(another domain)
* Reference-guided LPIPS values for all style: 0.4380792519592103
* FID between style and stylized: 127.29334051784087
* SIFID between style and stylized: 0.08613132
