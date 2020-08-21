# SSL-VQA
Here is the implementation of our IJCAI 2020 paper [Overcoming Language Priors with Self-supervised Learning for Visual Question Answering](https://www.ijcai.org/Proceedings/2020/0151.pdf). This repository contains code modified from [here](https://github.com/jialinwu17/self_critical_vqa), many thanks!

## Requirements
* python 3.6.8

* pytorch 1.0.1 

* zarr

* tdqm

* spacy

* h5py

## Download and preprocess the data

```
cd data 
bash download.sh
python preprocess_image.py --data trainval
python create_dictionary.py --dataroot vqacp2/
python preprocess_text.py --dataroot vqacp2/ --version v2
cd ..
```

## Training
* Train our model with multi-label VQA loss
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ 
--img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 3 --ml_loss
```
* Train our model with corss-entropy VQA loss
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ 
--img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 1.2 --ce_loss
```
* Train the model with 80% of the original training set
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ 
--img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 3 --ml_loss --ratio 0.8
```

## Evaluation
* A json file of results from the test set can be produced with:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot data/vqacp2/ --img_root data/coco/ --checkpoint_path saved_models_cp2/best_model.pth --output saved_models_cp2/result/
```
* Compute detailed accuracy for each answer type:
```
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/
```

## Pretrained model & Well-trained model
If you don't want to train from scratch, you can download the pretrained base model from [here](https://drive.google.com/file/d/1QRYHnmZ7Epauf-ooUkVL4IiQliRpny2c/view?usp=sharing)(for ml_loss), and fine-tune it with our self-supervised loss as below:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ 
--img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 3 --ml_loss --checkpoint_path ml_pretrained.pth
```
A well-trained model (for ml_loss) can be found [here](https://drive.google.com/file/d/1s9Q-26uNooXXLyRLF3-vfY1brvGf-Zia/view?usp=sharing). The test results file produced by it can be found [here](https://drive.google.com/file/d/1MXJ94BaFyhAOD2yTN1ROUim4vQsDEc1M/view?usp=sharing) and its performance is as follows:
```
Overall score: 58.58
Yes/No: 87.47 Num: 40.3 other: 48.45
```


## Reference
If you found this code is useful, please cite the following paper:
```
@inproceedings{ijcai2020-151,
  title     = {Overcoming Language Priors with Self-supervised Learning for Visual Question Answering},
  author    = {Zhu, Xi and Mao, Zhendong and Liu, Chunxiao and Zhang, Peng and Wang, Bin and Zhang, Yongdong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {1083--1089},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/151},
  url       = {https://doi.org/10.24963/ijcai.2020/151},
}
```


