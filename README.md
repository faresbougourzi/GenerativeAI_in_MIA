# Generative AI in Midical Image Analysis


## Contents
- [Related Surveys](#related-surveys)
- [Generative AI](#generative-ai)
  - [Generative Adversarial Nets (GANs)](#generative-adversarial-nets-gans)
  - [Diffusion Models](#anomaly-detection)
  - [Variational Auto-encoders](#anomaly-detection)
  - [ChatGPT-4 And LLMs](#anomaly-detection)
- [Medical Image Analysis](#anomaly-detection)
- [GANs in MIA](#gans-in-mia)
  - [Image-to-Image Translation](#image-to-image-translation)
  - [Anomaly Detection](#anomaly-detection)
  - [Reconstruction](#reconstruction)
  - [Classification](#classification)
  - [Segmentation](#segmentation)
  - [Others](#others1) 
- [Diffusion Models in MIA](#diffusion-models-in-mia)
  - [Image-to-Image Translation](#image-to-image-translation1)
  - [Anomaly Detection](#anomaly-detection)
  - [Reconstruction](#reconstruction)
  - [Classification](#classification)
  - [Segmentation](#segmentation)
  - [Others](#others) 
- [Other Generative models in MIA](#anomaly-detection)
  - [Variational Auto-encoders](#anomaly-detection)
  - [Neural Radiance Field](#anomaly-detection)
  - [Auto-Regressive Models](#anomaly-detection)
  - [Normalizing Flow Models](#anomaly-detection) 
  - [Energy Based Models](#anomaly-detection)
  - [ChatGPT-4 And LLMs](#anomaly-detection)
  - [Hybrid Generative AI models](#anomaly-detection)



## Related Surveys

**A Comprehensive Review of Generative AI in Healthcare** \
*Yasin Shokrollahi, Sahar Yarmohammadtoosky, Matthew M. Nikahd, Pengfei Dong, Xianqi Li, Linxia Gu* \
[24th Jul., 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2310.00795)]

**Generative AI for Medical Imaging: extending the MONAI Framework** :fire: \
*Walter H. L. Pinaya, Mark S. Graham, Eric Kerfoot, Petru-Daniel Tudosiu, Jessica Dafflon, Virginia Fernandez, Pedro Sanchez, Julia Wolleb, Pedro F. da Costa, Ashay Patel, Hyungjin Chung, Can Zhao, Wei Peng, Zelong Liu, Xueyan Mei, Oeslle Lucena, Jong Chul Ye, Sotirios A. Tsaftaris, Prerna Dogra, Andrew Feng, Marc Modat, Parashkev Nachev, Sebastien Ourselin, M. Jorge Cardoso* \
[27th Jul., 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2307.15208)] [[Github](https://github.com/Project-MONAI/GenerativeModels)]

**Deep Learning Approaches for Data Augmentation in Medical Imaging: A Review** \
*Aghiles Kebaili, Jérôme Lapuyade-Lahorgue, Su Ruan* \
[24th Jul., 2023] [Journal of Imaging, 2023] \
[[Paper](https://arxiv.org/abs/2307.13125)]

**A Comprehensive Survey on Generative Diffusion Models for Structured Data** \
*Heejoon Koo, To Eun Kim* \
[7th Jun., 2023] [arXiv, 2023]  \
[[Paper](https://arxiv.org/abs/2306.04139)]

**Diffusion Models for Time Series Applications: A Survey** \
*Lequan Lin, Zhengkun Li, Ruikun Li, Xuliang Li, Junbin Gao* \
[1st May, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.00624)]

**A Comprehensive Survey on Knowledge Distillation of Diffusion Models** \
*Weijian Luo* \
[9th Apr., 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.04262)] 

**A Survey on Graph Diffusion Models: Generative AI in Science for Molecule, Protein and Material** \
*Mengchun Zhang, Maryam Qamar, Taegoo Kang, Yuna Jung, Chenshuang Zhang, Sung-Ho Bae, Chaoning Zhang* \
[4th Apr., 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.01565)] 

**Audio Diffusion Model for Speech Synthesis: A Survey on Text To Speech and Speech Enhancement in Generative AI** \
*Chenshuang Zhang, Chaoning Zhang, Sheng Zheng, Mengchun Zhang, Maryam Qamar, Sung-Ho Bae, In So Kweon* \
[23th Mar., 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2303.13336)] 



<!--- Theory --->
## Generative AI

### Generative Adversarial Nets (GANs)

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Diffusion Models

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### ChatGPT-4 And LLMs

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]



<!--- MIA Introduction --->
## Medical Image Analysis

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]



<!--- GANs --->
## GANs in MIA

### Image-to-Image Translation

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Anomaly Detection

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Reconstruction

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Classification

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Segmentation

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Others

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]




<!--- Diffusion Models --->
## Diffusion Models in MIA

### Image-to-Image Translation(#image-to-image-translation1)

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Anomaly Detection

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Reconstruction

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Classification

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Segmentation

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Others

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]








<!--- Other Generative AI Models --->
## Other Generative AI Models in MIA

### Variational Auto-encoders

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Neural Radiance Field

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Auto-Regressive Models

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Normalizing Flow Models
**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### Energy Based Models

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]


### ChatGPT-4 And LLMs

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]

### Hybrid Generative AI models

**Modality Cycles with Masked Conditional Diffusion for Unsupervised Anomaly Segmentation in MRI** \
*Ziyun Liang, Harry Anthony, Felix Wagner, Konstantinos Kamnitsas* \
[30th Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.16150)] [[Github](https://github.com/ZiyunLiang/MMCCD)]

**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** \
*Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey* \
[3rd Aug, 2023] [arXiv, 2023]<br>
[[Paper](https://arxiv.org/abs/2308.02062)] [[Github](https://github.com/alessandro-f/Dif-fuse)]



# My GitHub Project

## Summary

This is the summary of my project.

## Table of Contents

- [Subtitle 1: Image to Image Translation](#image-to-image-translation-1)
- [Subtitle 2: Image to Image Translation](#image-to-image-translation-2)

## Subtitle 1: Image to Image Translation

This is the content for the first subtitle.

## Subtitle 2: Image to Image Translation

This is the content for the second subtitle.



