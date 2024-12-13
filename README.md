# MCP
<<<<<<< HEAD

PyTorch and tensorflow implementataion for the TCSVT 2024 paper: Mitigating Cross-modal Retrieval Violations with
Privacy-preserving Backdoor Learning

## Environment

Our dependency (Python 3.7, tensorflow 2.7.0, CUDA Version 11.6)

* you can 
```sh
pip install -r requirements.txt
```
* or download an entire package of environment and unzip it to conda's env folder, then just load it.

https://pan.baidu.com/s/1tkOPbWnX8QdnDipZtAZJyQ password: kcd3 


## Usage

**Train/test clean model**

```sh
python main.py:
  ----flag: fvc #dataset fvc:fashionvc, mir:flickr25k, nus:nus-wide
  --group: clean #The folder name for storing evaluation results
```
* For example

```sh
python main.py train --flag 'fvc' --group 'clean' #train
python main.py test --flag 'fvc' --group 'clean' #test
```

* Also, you can directly download the pre-training weight to test.
https://pan.baidu.com/s/1p_YYxVmROmvArMKqLVhVJA password: 7gud
(stored in checkpoints/dataset_name/bit_size/group_name) 

**Train/test LBL model**

```sh
python backdoor_main.py:
  --flag: fvc #dataset fvc:fashionvc, mir:flickr25k, nus:nus-wide
  --group: clean #The folder name for storing evaluation results
  --backdoor: True #The flag for LBL attack.
  --backdoor_trigger: StegaStamp #backdoor trigger,<BadNets, Blended, WavNet, StegaStamp(default)>
  --pr: 0.1 #poisoned rate, default:0.1
```

* For example

```sh
python backdoor_main.py train --flag 'fvc' --backdoor True --group 'StegaStamp_label' --backdoor_trigger 'StegaStamp' --pr 0.1 #train
python backdoor_main.py test --flag 'fvc' --group 'StegaStamp_label' --backdoor_trigger 'StegaStamp' #test
```

* Also, you can directly download the pre-training weight to test.
https://pan.baidu.com/s/1Ybdhc61BeMZjAsTqZ7megg password: z9uj

**Train/test MCP model**

```sh
python backdoor_main_lg.py:
  --flag: fvc #dataset fvc:fashionvc, mir:flickr25k, nus:nus-wide
  --group: clean #The folder name for storing evaluation results
  --backdoor_loss: True #The flag for MCP attack.
  --backdoor_trigger: StegaStamp #backdoor trigger,<BadNets, Blended, WavNet, StegaStamp(default)>
```
* For example

```sh
python backdoor_main_lg.py train --flag 'fvc' --backdoor_loss True --group 'StegaStamp_loss_lg' --backdoor_trigger 'StegaStamp' #train
python backdoor_main_lg.py test --flag 'fvc' --group 'StegaStamp_loss_lg' --backdoor_trigger 'StegaStamp' #test
```

* Also, you can directly download the pre-training weight to test:
https://pan.baidu.com/s/1d72uQmjFjCn0bCzUDI_XeA password: 4ut7


## Dataset

`FashionVC: image.mat, tag.mat, label.mat`
* link: https://pan.baidu.com/s/1lBB5taiPEdO8-iCAxzRU5Q?pwd=u4uk password: u4uk

`FLICKR-25K.mat` 
* link: https://pan.baidu.com/s/1IfQSG8ZRRidw1lQxdspJCw?pwd=i5wg  password: i5wg

`NUS-WIDE.mat`
* link: https://pan.baidu.com/s/1dZI_BvK-TAGJuXIvhUihTw?pwd=rprr  password: rprr

`imagenet-vgg-f.mat` 
* link: https://pan.baidu.com/s/1RsI8FtqqxGBpzLm7eNVE8g?pwd=17kt  password: 17kt

> Dataset source and Pytorch&TensorFlow souce code：https://github.com/lqsunshine/MCP

note: remember to change backdoor/init.py/model_path to the path of pre-training weight of StegaStamp's encoder. 
=======
This is the official implementation of our paper "Mitigating Cross-modal Retrieval Violations with Privacy-preserving
Backdoor Learning". This research project is developed based on Python 3, Pytorch and TensorFlow, created by Qiang Liu.

The codes will be released upon the acceptance of this paper.
>>>>>>> 5a119e4e47c1032371711a170717794b3f2948bd
