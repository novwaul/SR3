# SR3
<p> Reimplementation of 4x SR3 https://arxiv.org/abs/2104.07636 </p>
<p> The UNet structure is almost same as vanilla DDPM except that self-attention is performed at the last depth and the depth right before the last depth, and group normalization is performed on total 8 groups instead of 32 groups. As mentioned in the paper, gamma value is sampled between two alpha values at t-1 and t with unifrom probability distribution, and the value is directly inserted to embedding generation module just like time value in DDPM. </p>

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
|Sample Beta Schedule|Linear Schedule from 1e-4 to 0.1|
|Train Steps|1000|
|Sample Steps|100|

#### B. Scores
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 64x64 to 256x256|(12.829, 0.992)|3.642|23.185|0.564|

#### C. Samples

##### Validation
|Tag|Image|
|:---:|:---:|
|LR|![LR64_val](https://github.com/novwaul/SR3/assets/53179332/f7e3974f-d503-43d1-9a13-3fe4ee2e8d0c)|
|Sample|![Sample64_val](https://github.com/novwaul/SR3/assets/53179332/70dba161-3b20-472d-b4b5-0dcc0748d657)|
|HR|![HR64_val](https://github.com/novwaul/SR3/assets/53179332/ca2736aa-e350-4a81-bdf6-6abb8313a55d)|

##### Test
|Tag|Image|
|:---:|:---:|
|LR|![LR64](https://github.com/novwaul/SR3/assets/53179332/656a7d4b-1925-42b8-b74b-698b13ec98ff)|
|Sample|![Sample64](https://github.com/novwaul/SR3/assets/53179332/5a922a74-2770-4b5c-8ca6-aeb2a7ddd3f7)|
|HR|![HR64](https://github.com/novwaul/SR3/assets/53179332/c8e53193-4c86-4caf-aa79-d9d314a5a9c3)|

### 32x32 to 128x128 Model
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 32x32 to 128x128|(-, -)|-|-|-|

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
|Sample Beta Schedule|Linear Schedule from 1e-6 to 0.05|
|Train Steps|1000|
|Sample Steps|100|
