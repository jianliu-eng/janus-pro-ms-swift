## 为了方便安装直接使用如下命令：
```bash
pip install git+https://github.com/deepseek-ai/Janus
```



:lantern: 为了便于下载模型文件，可以使用model_scope

```
pip install modelscope
```

[参考了modelscope社区上的案例](https://mp.weixin.qq.com/s/TjeY6bHk6WhBcb6K6GfZLQ)

#### :hamburger: 加载deepseek- Janus-7B多模态模型

```
# todo 图片理解

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from modelscope import snapshot_download

# specify the path to the model
# model_path = snapshot_download("deepseek-ai/Janus-Pro-7B", local_dir="./model")
model_path = "/data/ms-swift/output/v2-20250222-140230/checkpoint-3"

print("模型路径：", model_path)

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

question = "描述一下图片内容"

image = "./data/images/mapo-tofu.png"
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

下面开始微调（图像理解的微调训练） ms-swift

1. 准备环境 :fire_engine:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swiftc
pip install -e .
```



2. 微调命令：

   微调工具使用:tomato:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /data/janus-sft/model/deepseek-ai/Janus-Pro-7B \
    --dataset AI-ModelScope/LaTeX_OCR:human_handwrite#50 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 
```

   当然可以预览数据集，访问如下地址：

[点击访问](https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary)

或者可以使用如下python代码加载数据集，进行预览：

```python
from modelscope import MsDataset
train_dataset = MsDataset.load("AI-ModelScope/LaTeX_OCR", subset_name="small", split="train")
train_dataset[2]
#{'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=200x50 at 0x15A5D6CE210>,
#'text': '\\rho _ { L } ( q ) = \\sum _ { m = 1 } ^ { L } \\ P _ { L } ( m ) \\ { \\frac { 1 } { q ^ { m - 1 } #} } .'}
len(train_dataset)
#50

```
3. 推理

 ```bash
   CUDA_VISIBLE_DEVICES=0 \
   swift infer \
       --adapters /data/ms-swift/output/v4-20250223-124941/checkpoint-3 \
       --stream false \
       --max_batch_size 1 \
       --load_data_args true \
       --max_new_tokens 2048
 ```

4. 保存模型--(推送到modelscope)

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters /data/ms-swift/output/v4-20250223-124941/checkpoint-3 \
    --push_to_hub true \
    --hub_model_id '你的模型id' \
    --hub_token '你的token'   
```

   



