# SR3
<p> Reimplementation of 4x SR3 https://arxiv.org/abs/2104.07636 </p>


## Result
For all evaluation, Flickr2K 1k samples (from img 1 to img 1000) are used.

#### 64x64 to 256x256 Model
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 64x64 to 256x256|(12.632, 0.982)|3.745|23.191|0.564|

#### 32x32 to 128x128 Model
|Dataset|IS (Mean, Std.)|FID|PSNR|SSIM|
|:---:|:---:|:---:|:---:|:---:|
|centor crop 32x32 to 128x128|(5.441, 0.547)|8.088|29.549|0.695|
