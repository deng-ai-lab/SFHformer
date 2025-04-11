## [When Fast Fourier Transform Meets Transformer for Image Restoration] (ECCV 2024)
 Official implementation.
 
## Authors
Xingyu Jiang, Xiuhui Zhang, Ning Gao, Yue Deng *

School of Astronautics, Beihang University, Beijing, China
 
#### News
Thanks for your interest in our work, we will continue to optimize our code. If you have any other questions, please feel free to raise them in the issues, and I will try my best to address them!
- **Apr 11, 2025:** We release some visualizations of the dataset in the Visual result section. 
- **Mar 27, 2025:** We release the pre-training weights of ITS and OTS with the test code in the dehazing folder.
- **Oct 17, 2024:** The train code is now open and our paper is available [here](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06190.pdf)! 
- **Jul 25, 2024:** Paper accepted at ECCV 2024.

<hr />

> **Abstract:** *Natural images can suffer from various degradation phenomena caused by adverse atmospheric conditions or unique degradation mechanism. Such diversity makes it challenging to design a universal framework for kinds of restoration tasks. Instead of exploring the commonality across different degradation phenomena, existing image restoration methods focus on the modification of network architecture under limited restoration priors. In this work, we first review various degradation phenomena from a frequency perspective as prior. Based on this, we propose an efficient image restoration framework, dubbed SFHformer, which incorporates the Fast Fourier Transform mechanism into Transformer architecture. Specifically,  we design a dual domain hybrid structure for multi-scale receptive fields modeling, in which the spatial domain and the frequency domain focuses on local modeling and global modeling, respectively. Moreover, we design unique positional coding and frequency dynamic convolution for each frequency component to extract rich frequency-domain features. Extensive experiments on thirty-one restoration datasets for a range of ten restoration tasks such as deraining, dehazing, deblurring, desnowing, denoising, super-resolution and underwater/low-light enhancement, demonstrate that our SFHformer surpasses the state-of-the-art approaches and achieves a favorable trade-off between performance, parameter size and computational cost.* 
<hr />

## Introduction
<p align='center'>
<img src = "image/introduce.jpg"> 

## Network Architecture
<p align='center'>
<img src = "image/method.jpg"> 


## Results
Experiments are performed for different image restoration tasks including, image dehazing, image deraining, image desnowing, image denoising, image super-resolution, single-image motion deblurring, defocus deblurring, image raindrop removal, low-light image enhancement and underwater image enhancement. 

<details>
<summary><strong>Image Dehazing</strong> (click to expand) </summary>
<p align='center'>
<img src = "image/haze_result.png"> 
</details>

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>
<p align='center'>
<img src = "image/rain_result.png"> 
</details>

<details>
<summary><strong>Image Desnowing</strong> (click to expand) </summary>
<p align='center'>
<img src = "image/snow.png"> 
</details>

<details>
<summary><strong>Image Super-resolution</strong> (click to expand) </summary>
<p align='center'>
<img src = "image/supp_super.png"> 
</details>

<details>
<summary><strong>Image Raindrop Removal</strong> (click to expand) </summary>
<p align='center'>
<img src = "image/raindrop.png"> 
</details>

<details>
<summary><strong>Single-Image Motion Deblurring</strong> (click to expand) </summary>
<p align="center">
<img src = "image/motion_deblur.png" >
</details>

<details>
<summary><strong>Defocus Deblurring</strong> (click to expand) </summary>
<p align="center">
<img src = "image/supp_defocus.png"> 
</details>


<details>
<summary><strong>Image Denoising</strong> (click to expand) </summary>
<p align="center">
<img src = "image/supp_noise.png"> 
</details>

<details>
<summary><strong>Underwater Image Enhancement</strong> (click to expand) </summary>
<p align="center">
<img src = "image/underwater_result.png"> 
</details>

<details>
<summary><strong>Low-light Image Enhancement</strong> (click to expand) </summary>
<p align="center">
<img src = "image/lowlight.png"> 
</details>


## Visual Results
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>SOTS-indoor</th>
    <th>SOTS-outdoor</th>
    <th>O-HAZE</th>
    <th>NH-HAZE</th>
    <th>DENSE-HAZE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/1w3GhWD5yAd8N_JsJoXBtXA?pwd=8sj6">Download (8sj6)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1A5B3hm39YrB51rvX3RWtVw?pwd=awnk">Download (awnk)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1OMJDVsJoh4zrlvDWlZ2FNA?pwd=pfem">Download (pfem)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1N75cBd3GIinW6NA_WxiRSg?pwd=e72s">Download (e72s)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1qG3IjMm-sOnPepmyZvEoQg?pwd=y8we">Download (y8we)</a>  </td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>LOLv2-real</th>
    <th>LOLV2-syn</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/1PvmOpyZEfvZ3BFvJYqaMfw?pwd=jqgh">Download (jqgh)</a>  </td>
    <td> <a href="https://pan.baidu.com/s/1Ub4xxkVXft9cKw2dkR1Rgg?pwd=wy8i">Download (wy8i)</a>  </td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>GoPro</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/1QKJk9BTxP0GI5yyferFG5w?pwd=z9uv">Download (z9uv)</a>  </td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>SRRS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/1MAAIIOjWW_5JhzqQfakR0A?pwd=5899">Download (5899)</a>  </td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>RainDrop</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/1jSGfmuNaPMXGweFEJO-D5g?pwd=4nay">Download (4nay)</a>  </td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu NetDisk</td>
    <td> <a href="https://pan.baidu.com/s/11BsZhTnOmJPpDlgzQuThJQ?pwd=k8s6">Download (k8s6)</a>  </td>
  </tr>
</tbody>
</table>
