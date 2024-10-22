# OpenAI API Demo

使用教程：https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/openai_api_zh

Tutorial: https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/openai_api_en

# 本地部署Llama-3-Chinese并使用chatnext进行测试
### 介绍

大模型项目网址：（https://github.com/ymcui/Chinese-LLaMA-Alpaca-3）

本地部署大模型，并封装一个私有的兼容openai api的大模型接口，使用ChatGPTNextWeb调用此接口，需要准备单张显存大于20GB的显卡，开量化后只需要11GB。

### 环境搭建

#### linux服务器环境

创建虚拟环境

```shell
conda create -n llama python=3.8.17 pip -y
```

进入虚拟环境，安装modelscopy（国内下载模型的工具），并移动到autodl-tmp目录下

```
conda activate llama
pip install modelscope -U
modelscope download --model ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3
mv /root/.cache/modelscope/hub/ChineseAlpacaGroup /root/autodl-tmp/.autodl
```

默认存储在了~/.cache/modelscope/hub目录下，可以修改sgsdfgsdfgsdfgsfd

下载源代码并进入目录下

```
wget https://file.huishiwei.top/Chinese-LLaMA-Alpaca-3-3.0.tar.gz
tar -xvf Chinese*
cd Chinese-LLaMA-Alpaca-3-3.0
```

安装对应环境

```
pip install -r requirements.txt
pip install fastapi sse_starlette shortuuid pydantic==1.10.11
```

pydantic必须为1.10.11，否则报下面的错误
![image-20241021100449419](https://github.com/user-attachments/assets/f3ca0afe-c259-437b-ba64-85c8c4814cce)

为了兼容llama-3-Chinese模型，需要修改scripts/oai_api_demo/openai_api_server.py文件中下面的generation_kwargs，需要添加pad_token_id和eos_token_id，否则没有输出终止符会不断的输出。

```
generation_kwargs = dict(
    streamer=streamer,
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=False,
    max_new_tokens=max_new_tokens,
    repetition_penalty=float(repetition_penalty),
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)
```

#### windows测试环境

下载ChatGPTNextWeb，下载地址：[Release v2.15.5 Google Gemini support function call · ChatGPTNextWeb/ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web/releases/tag/v2.15.5)



### 测试

linux服务器启动服务，如果需要开启量化，可以加上--load_in_8bit或者--load_in_4bit，如果只想用cpu加载，加上--only_cpu参数

```shell
python openai_api_server.py --gpus 0 --base_model /root/autodl-tmp/.autodl/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3/
```

显示下面内容表示开启成功，开放端口为19327
![image-20241021102219239](https://github.com/user-attachments/assets/cba9f9c7-a483-452b-be76-1eebb02c5d02)

再开启一个窗口，执行下面命令每一秒中更新并查看显存占用空间

```shell
watch -n 1 nvidia-smi
```

推理阶段显存基本保持不变，但是功耗会上升，下面是不同加载方式的显存占用

|      |  显存   | 推理速度 |
| :--: | :-----: | :------: |
| 全量 | 15708MB |   很快   |
| 8bit | 9336MB  |    慢    |
| 4bit | 5744MB  |   中等   |
| cpu  |    0    |  非常慢  |

由于autodl不对外开放这个端口，windows本地访问连接不上，需要在windows上开启一个ssh通道，执行下面命令，表示本地开放的端口19327会连接到远程服务器的19327端口

```shell
ssh -CNgv -L 19327:127.0.0.1:19327 root@connect.beijinga.seetacloud.com -p 54377
```

执行完之后，要求输入密码，之后就能一直监听系统了
![image-20241021102523954](https://github.com/user-attachments/assets/4facd0e6-eef6-4ef3-9236-d354526841f6)

