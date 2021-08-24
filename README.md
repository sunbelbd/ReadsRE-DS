# ReadsRE: Retrieval-Augmented Distantly Supervised Relation Extraction, SIGIR 21
This repository contains PaddlePaddle code that supports experiments in our SIGIR 2021 paper: ReadsRE: Retrieval-Augmented Distantly Supervised Relation Extraction. Note: Pytorch version is available upon request. 

## Environment: refer to requirements.txt

## Dataset Preparation

NYT10 Dataset Downloads;

`mkdir nyt10`
`wget -P nyt10 https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_rel2id.json`
`wget -P nyt10 https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_train.txt`
`wget -P nyt10 https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_test.txt`

Wikipedia Dataset Downloads:

`wget https://archive.org/download/enwiki-20190201/enwiki-20190201-pages-articles-multistream.xml.bz2`


## Single-GPU Version

```cd paddle_exp```
```python main.py```


## Multi-gpu Version

`cd paddle_dist`
`python3 -m paddle.distributed.launch --gpus=4,6 main_dist.py`


## Reference
If you find our code or work useful, please cite it as below:
```
@inproceedings{ZhangSIGIR21,
  author    = {Yue Zhang and Hongliang Fei and Ping Li},  
  title     = {ReadsRE: Retrieval-Augmented Distantly Supervised Relation Extraction},
  booktitle = {{SIGIR} '21: The 44th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Virtual Event, Canada, July
               11-15, 2021},
  pages     = {2257--2262},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3404835.3463103},
  doi       = {10.1145/3404835.3463103},
  timestamp = {Thu, 15 Jul 2021 15:30:48 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/ZhangF021.bib},
}
```
