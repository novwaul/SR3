# SR3
<p> Reimplementation of 4x SR3 https://arxiv.org/abs/2104.07636 </p>


## Result

### 64x64 to 256x256 Model

#### A. Settings
|Tag|Setting|
|:---:|:---:|
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
|LR|![LR64_val](https://github.com/novwaul/SR3/assets/53179332/d67c1f49-e92c-40f1-8333-6358eb5781a6)|
|Sample|![Sample64_val](https://github.com/novwaul/SR3/assets/53179332/717ed6f6-b5d1-4de2-838d-48aeaef68b34)|
|HR|![HR64_val](https://github.com/novwaul/SR3/assets/53179332/68a2c1f5-8465-4422-b2d2-79dbfa7b73cb)|

##### Test
|Tag|Image|
|:---:|:---:|
|LR|![LR64](https://github.com/novwaul/SR3/assets/53179332/3b0b886e-830a-49df-83c1-88a6065254c8)|
|Sample|![Sample64](https://github.com/novwaul/SR3/assets/53179332/86871fb6-e21e-4f90-b778-d6c723a9939a)|
|HR|![HR64](https://github.com/novwaul/SR3/assets/53179332/90d8e3d5-abeb-436a-96a5-f0f764894cdb)|


### 32x32 to 128x128 Model
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 32x32 to 128x128|(5.441, 0.547)|8.088|29.549|0.695|
