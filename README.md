This a framework about DeepLearning Project.

Give an example about [cycle-gan](https://junyanz.github.io/CycleGAN/) 
and [pix2pix](https://phillipi.github.io/pix2pix/).

## Project Structure
```
├── data
│   ├── __init__.py
│   ├── image_folder.py
│   ├── base_dataset.py
│   ├── aligned_dataset.py
│   ├── unaligned_dataset.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── cycle_gan_model.py
│   ├── pix2pix_model.py
├── options
│   ├── __init__.py
│   ├── base_options.py
│   ├── train_options.py
├── util
│   ├── __init__.py
│   ├── logger.py
│   ├── util.py
│   ├── image_pool.py
│   ├── visualizer.py
├── train.py
├── trainer.py
```

## Requirements
- Python 3.11
- PyTorch 2.0.0
- torchvision


## Usage

#### CycleGAN

Image input size: (256, 256)

Download dataset
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

Train
```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan
```

#### Pix2Pix

Image input size: (512, 256)

Download dataset
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

Train
```bash
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --batch_size 1024
```


---
## Acknowledgments
This project is inspired by [cycle-gan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Thanks for the author's contribution.
