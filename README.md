# 聊天机器人

## 一、简介

基于Transformer模型构建的聊天机器人，可实现日常聊天。

## 二、系统说明

### 2.1 功能介绍

使用者输入文本后，系统可根据文本做出相应的回答。

### 2.2 数据介绍

* 百度中文问答 WebQA数据集
* 青云数据集
* 豆瓣数据集
* chatterbot数据集

由于数据集过大，因此不会上传，如有需要可以在issue中提出。

### 2.3. 模型介绍（v1.0版本）

基于Transformer模型，使用Python中的keras-transformer包。

训练的参数文件没有上传，如有需要可在issue中提出。

## 三、注意事项

* keras-transformer包需要自行安装：`pip install keras-transformer`。
* 如果需要实际运行，可在issue中提出，并将我提供的参数文件放在`ModelTrainedParameters`文件下；`ListData`文件下包含了已经处理好的字典等数据，不需要修改，直接运行Main.py即可。
* 如果需要自行训练，可在issue中提出，并将数据集文件放在`DataSet`文件下。
* `HyperParameters.py`文件中包含了系统所需要的超参数，包括文件路径等，可根据需要自行修改；其中包含了训练模型、重新训练模型、测试模型（实际运行）的控制参数，可自行修改使用。

