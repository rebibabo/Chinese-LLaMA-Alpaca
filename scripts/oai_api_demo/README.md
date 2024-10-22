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

##### 使用ChatNext测试

windows打开NextChat，点开设置
![image-20241021102103855](https://github.com/user-attachments/assets/5b94a08c-fdbf-479e-b014-f0bc408089c3)

![image-20241021103012907](https://github.com/user-attachments/assets/59c855ea-c937-4957-b016-923209c1e8f1)

![image-20241021103058828](https://github.com/user-attachments/assets/e7fe5c93-11de-4418-b5df-058595552234)

##### 使用命令测试

```shell
curl http://127.0.0.1:19327/v1/chat/completions  \
-H"Content-Type: application/json" \
-H"Authorization: Bearer x" -d'{
    "model":"llama-3-chinese",
    "messages":[
        {
            "role":"user",
            "content":"1+1等于几?"
        }
    ],
    "max_tokens": 4096
}'
```

##### 使用python代码测试

```python
import requests
import json

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer x"
}
url = "http://127.0.0.1:19327/v1/chat/completions"
data = {
    "model": "llama-3-chinese",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ],
    "max_tokens": 4096
}
data = json.dumps(data, separators=(',', ':'))
response = requests.post(url, headers=headers, data=data)

print(response.text)
print(response)
```

### GPU释放问题

当模型推理时，功耗可以达到340W左右，生成完后的十几秒内，功耗在100W左右，接着降低到20W左右，点击暂停键并不能停止生成，功耗仍然很高。
![image-20241021110422896](https://github.com/user-attachments/assets/56af3dcf-d655-4b87-9567-e8f4ce0d1250)
现在能够当接收到停止的时候，立即停止生成，在server代码中开头添加需要的库，并创建一个停止原则类

```python
import asyncio
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnEvent(StoppingCriteria):
    def __init__(self) -> None:
        self.is_stop = False
        super().__init__()
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.is_stop:
            return True
        return False
```

接着新建立一个函数event_publisher来替代原来的stream_predict函数，和之前的不同的是，当接收到asyncio.CancelledError事件时，设置StopOnEvent的is_stop为True，并将其传到generation_kwargs中的stopping_criteria参数。

```python
nc def event_publisher(
    input,
    max_new_tokens=4096,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="chinese-llama-alpaca-2",
    **kwargs,
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    
    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stop = StopOnEvent()
    generation_kwargs = dict(
        streamer=streamer,
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(repetition_penalty),
        stopping_criteria=StoppingCriteriaList([stop]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    i = 0
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    try:
        for new_text in streamer:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                choices=[choice_data],
                object="chat.completion.chunk"
            )
            chunk_str = "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
            print(chunk_str)
            i += 1
            yield chunk_str
            await asyncio.sleep(0.03)
    except asyncio.CancelledError as e:
        print(f"Disconnected from client (via refresh/close)")
        stop.is_stop = True
        raise e
```

但是测试的时候，发现点击停止并没有进入`except asyncio.CancelledError as e`



# 使用oneapi网关实现大模型接口高可用方案

### 介绍

基于docker安装oneapi，连接已经部署在不同网络下的大模型，并共用同一个端口。

一共三台主机，一台服务器，为AutoDL租赁的，用于大模型推理的，一台windows上的Ubuntu虚拟机，需要下载docker并搭建One API网关，用于分配端口到各个大模型所在服务器，并对外开放唯一一个端口，一台windows主机，用于访问One API网关。

### 环境搭建

服务器上确保部署好了大模型，并开辟两条进程，分别对应gpu版本和cpu版本。

在server.py文件中，添加一个新的参数port，默认为19327

```python
parser.add_argument('--port', default=19327, type=int)
```

修改uvicorn.run中的port参数为args.port

```python
uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1, log_config=log_config)
```

启动gpu版本，端口号为默认的19327

```shell
python openai_api_server.py --gpus 0 --base_model /root/autodl-tmp/.autodl/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3/
```

启动cpu版本，端口号设置为19328

```shell
python openai_api_server.py --only_cpu --base_model /root/autodl-tmp/.autodl/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3/ --port 19328
```



接下来是网关服务器（linux虚拟机上的docker）环境部署。

在Windows虚拟机上开辟通道，连接AutoDL租赁的服务器，并使用curl测试，能够获取结果。

```shell
ssh -CNgv -L 19327:127.0.0.1:19327 root@connect.beijinga.seetacloud.com -p 54377
curl http://127.0.0.1:19327/v1/chat/completions  \
-H"Content-Type: application/json" \
-H"Authorization: Bearer x" -d'{
    "model":"llama-3-chinese",
    "messages":[
        {
            "role":"user",
            "content":"1+1等于几?"
        }
    ],
    "max_tokens": 4096
}'
```

```shell
ssh -CNgv -L 19328:127.0.0.1:19328 root@connect.beijinga.seetacloud.com -p 54377
curl http://127.0.0.1:19328/v1/chat/completions  \
-H"Content-Type: application/json" \
-H"Authorization: Bearer x" -d'{
    "model":"llama-3-chinese",
    "messages":[
        {
            "role":"user",
            "content":"1+1等于几?"
        }
    ],
    "max_tokens": 4096
}'
```



如果之前安装了docker，要卸载掉。

```shell
sudo apt remove docker-ce docker-ce-cli containerd.io docker-compose-plugin docker docker-engine docker.io containerd runc
```

首先换apt源，否则安装不了，编辑/etc/apt/sources.list，添加任意一个国内源

```shell
#中科大
deb http://mirrors.ustc.edu.cn/kali kali-rolling main non-free contrib
deb-src http://mirrors.ustc.edu.cn/kali kali-rolling main non-free contrib
 
#阿里云
deb http://mirrors.aliyun.com/kali kali-rolling main non-free contrib
deb-src http://mirrors.aliyun.com/kali kali-rolling main non-free contrib
 
#清华大学
deb http://mirrors.tuna.tsinghua.edu.cn/kali kali-rolling main contrib non-free
deb-src https://mirrors.tuna.tsinghua.edu.cn/kali kali-rolling main contrib non-free
 
#浙大
deb http://mirrors.zju.edu.cn/kali kali-rolling main contrib non-free
deb-src http://mirrors.zju.edu.cn/kali kali-rolling main contrib non-free

```

不要用公司内网下载，否则报错要网络认证
![image-20241021134448893](https://github.com/user-attachments/assets/65fd3da4-8535-4962-9b3f-48775716867b)
安装docker命令

```shell
# 安装ca-certificates curl gnupg lsb-release
sudo apt install ca-certificates curl gnupg lsb-release -y
#下载 Docker 的 GPG 密钥，并将其添加到 apt-key 中
sudo curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
# 安装软件包列表
sudo apt-get install packagekit 
# 为Docker 添加阿里源
sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
# 更新系统的软件包
sudo apt -y update
# 安装docker相关的包
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
```

创建docker文件夹，并创建docker-compose文件

```shell
mkdir -p ~/oneapi-compose
cd ~/oneapi-compose
touch docker-compose.yaml
```

向文件中添加下面内容

```yaml
version: '3.8'
services:
  oneapi:
    image: m.daocloud.io/ghcr.io/songquanpeng/one-api:v0.6.7
    container_name: oneapi
    restart: always
    ports:
      - 3030:3000
    networks:
      - oneapi_llm_course
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - ./data:/data
networks:
  oneapi_llm_course:
```

启动容器

```shell
docker compose up -d
```

打开火狐浏览器，输入127.0.0.1:3030，打开OneAPI网页
![image-20241021145335580](https://github.com/user-attachments/assets/a6d0363f-28a8-4260-acf4-32d8416c9578)
点击登录，用户名root，默认密码123456，登陆进去需要修改密码
![image-20241021151657297](https://github.com/user-attachments/assets/992781d6-b44b-4f9d-b208-58fe73f469de)
然后点击渠道，点击添加新的渠道
![image-20241021153539279](https://github.com/user-attachments/assets/f61b8b9f-e708-4f41-ab6e-bc3109467e71)
按照步骤依次填写信息
![image-20241021154103493](https://github.com/user-attachments/assets/ae23d98c-f376-4a3b-acf2-47e7c6987387)
点击测试报错
![image-20241021163033914](https://github.com/user-attachments/assets/7506d706-3413-490f-9175-989e06907cab)
错误原因，地址应该填写虚拟机的IP地址，而不是127.0.0.1，因为Windows主机和One API网关位于同一局域网下，而One API网关向AutoDL服务器开辟了隧道，Windows主机先是要访问One API，由One API进行转发，地址是由One API的IP地址发送的，所以要填写IP地址，不能填写本机地址，不然发送不出去。

之后点击测试，可以看到连接成功了。
![image-20241022100301418](https://github.com/user-attachments/assets/62436e49-0a0d-48ef-b707-2d3ca5d5ce70)
接着点击令牌，添加新的令牌
![image-20241022100409660](https://github.com/user-attachments/assets/f03356be-6762-43e8-898b-a5a1feba9564)
在这里可以设置令牌信息，例如模型范围、过期时间、额度限制等
![image-20241022100515746](https://github.com/user-attachments/assets/14716d33-f6f2-4778-9b15-ddbb3aff3975)
点击复制Token，用于下面配置NextChat
![image-20241022100659746](https://github.com/user-attachments/assets/85aff809-549e-436d-a899-37f87ba718b2)
现在打开ChatNext，点击设置，修改接口地址和API Key
![image-20241022101130236](https://github.com/user-attachments/assets/daa56a20-2891-4683-9c7d-9e2da56d1ac1)
在服务端，在下面代码中添加打印chunk的代码，

```python
chunk_str = "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
print(chunk_str)
```

再次进入聊天窗口，可以正常聊天。
![image-20241022101436368](https://github.com/user-attachments/assets/c00d6ac5-f539-4bc0-8c7d-4e479d71195a)
查看服务器输出端，打印的chunk和NextChat打印的能对的上，说明连接成功。
![image-20241022101309160](https://github.com/user-attachments/assets/e40d2d77-c728-4f7c-8bdd-fbd23c5e65b9)
现在修改NextChat设置中的令牌，尝试再次了解，返回无效令牌的错误，说明令牌的设置起效果了。
![image-20241022101015736](https://github.com/user-attachments/assets/9b26ed55-77f9-4e4f-bfa7-ea3e5afaad34)
回到One API，现在任意禁用cpu端口。
![image-20241022101708993](https://github.com/user-attachments/assets/df596777-e64a-4eac-be60-54871a205925)
回到NextChat，仍然能够通过gpu聊天，明显感到速度快多了。

在One API处设置额度大小一个很小的值，再次聊天，收到令牌额度用尽的错误信息
![image-20241022101858641](https://github.com/user-attachments/assets/29ca59b3-7bc4-447f-82ee-43edad44541d)
