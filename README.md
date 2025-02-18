# SR3
<p> Reimplementation of 4x SR3 https://arxiv.org/abs/2104.07636 </p>
<p> The UNet structure is almost same as the vanilla DDPM, except that self-attention is performed at the last depth and the depth right before the last depth, group normalization is performed on total 8 groups instead of 32 groups, and the linear scale of embedding generation module is replaced from 10,000 to 5,000. As mentioned in the paper, gamma value is sampled between two alpha values at t-1 and t with a unifrom probability distribution, and the square rooted value of gamma is directly inserted to the embedding generation module.</p>

[2025.02.18 edit]
The below results were measured without self-attention due to the typo in the self-attention block. The typo is now fixed.

## Result

### 64x64 to 256x256 Model

#### A. Settings
|Tag|Setting|
|:---:|:---:|
|Base Channel|56|
|Train Batch Size|4|
|Train Iterations|500K|
|Trian Data|DIV2K Train Set + Flickr2K Train Set from 1001 to 2650 images|
|Validation Data|DIV2K Validation Set|
|Test Data|Flickr2K Train Set from 1 to 1000 images|
|Train Data Augmentation|Random Crop, Random Flip, Random Rotation|
|Test Data Augmentation|Centor Crop|
|Train Learning Rate Schedule|Cosine Annealing Schedule from 1e-5 to 1e-7|
|Train Beta Scehdule|Linear Schedule from 1e-4 to 0.005|
|Sample Gamma Schedule|Linear Schedule from 1e-4 to 0.1|
|Train Steps|1000|
|Sample Steps|100|

#### B. Scores
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 64x64 to 256x256|(12.829, 0.992)|3.642|23.185|0.564|
|centor crop 256x256 to 1024x1024|(21.305, 2.290)|0.312|23.819|0.617|

<p>Note that this model does not train on 256x256 to 1024x1024.</p>
<p>Inception Score shows low values as cropped images are hard to recognize as an object. As crop size increases, Inception Score also increases.</p>

#### C. Samples
<p>Note that the below LR images are upsampled images by using bicubic interpolation.</p>

##### Validation (64x64 to 256x256)
|Tag|Image|
|:---:|:---:|
|LR|![LR64_val](https://github.com/novwaul/SR3/assets/53179332/f7e3974f-d503-43d1-9a13-3fe4ee2e8d0c)|
|Sample|![Sample64_val](https://github.com/novwaul/SR3/assets/53179332/70dba161-3b20-472d-b4b5-0dcc0748d657)|
|HR|![HR64_val](https://github.com/novwaul/SR3/assets/53179332/ca2736aa-e350-4a81-bdf6-6abb8313a55d)|

##### Test (64x64 to 256x256)
|Tag|Image|
|:---:|:---:|
|LR|![LR64](https://github.com/novwaul/SR3/assets/53179332/656a7d4b-1925-42b8-b74b-698b13ec98ff)|
|Sample|![Sample64](https://github.com/novwaul/SR3/assets/53179332/5a922a74-2770-4b5c-8ca6-aeb2a7ddd3f7)|
|HR|![HR64](https://github.com/novwaul/SR3/assets/53179332/c8e53193-4c86-4caf-aa79-d9d314a5a9c3)|

##### Test (256x256 to 1024x1024)
|Tag|Image|
|:---:|:---:|
|LR|![LR256](https://github.com/novwaul/SR3/assets/53179332/41a1d329-9123-4e11-a03d-66d92a528241)|
|Sample|![Sample256](https://github.com/novwaul/SR3/assets/53179332/6cdccc42-5ba4-4294-bf16-8ecd11cca827)|
|HR|![HR256](https://github.com/novwaul/SR3/assets/53179332/6a13c426-79bd-4b5e-bc95-45ade689fff6)|


### 32x32 to 128x128 Model
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 32x32 to 128x128|(7.159, 0.437)|8.177|23.609|0.563|

#### A. Settings
|Tag|Setting|
|:---:|:---:|
|Base Channel|64|
|Train Batch Size|12|
|Train Iterations|500K|
|Trian Data|DIV2K Train Set + Flickr2K Train Set from 1001 to 2650 images|
|Validation Data|DIV2K Validation Set|
|Test Data|Flickr2K Train Set from 1 to 1000 images|
|Train Data Augmentation|Random Crop, Random Flip, Random Rotation|
|Test Data Augmentation|Centor Crop|
|Train Learning Rate Schedule|Cosine Annealing Schedule from 1e-5 to 1e-7|
|Train Beta Scehdule|Linear Schedule from 1e-4 to 0.005|
|Sample Gamma Schedule|Linear Schedule from 1e-6 to 0.05|
|Train Steps|1000|
|Sample Steps|100|

#### C. Samples
<p>Note that the below LR images are upsampled images by using bicubic interpolation.</p>

##### Validation (32x32 to 128x128)
|Tag|Image|
|:---:|:---:|
|LR|![LR-Val](https://github.com/novwaul/SR3/assets/53179332/935f9984-2da7-436b-90f4-87d4ac482267)|
|Sample|![Sample-Val](https://github.com/novwaul/SR3/assets/53179332/c759b42b-a6bb-48bd-8c10-dc0da2d4c104)|
|HR|![HR-Val](https://github.com/novwaul/SR3/assets/53179332/56744f03-edca-477b-88dc-2a0e1f4be808)|

##### Test (32x32 to 128x128)
|Tag|Image|
|:---:|:---:|
|LR|![LR](https://github.com/novwaul/SR3/assets/53179332/3ed9dfea-d9b0-4c5e-a311-61937457a9c5)|
|Sample|![Sample](https://github.com/novwaul/SR3/assets/53179332/6b9322b9-fb03-4d01-8422-81ef7d261c30)|
|HR|![HR](https://github.com/novwaul/SR3/assets/53179332/47e1a173-dcf5-445b-9472-cab554105e7a)|

