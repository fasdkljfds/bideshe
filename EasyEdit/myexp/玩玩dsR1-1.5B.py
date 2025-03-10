from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
)


def stream_completion(user_msg, max_tokens=1000, temperature=0.5):
    messages = [{"role": "user", "content": user_msg}]

    # 将对话格式转换为模型输入
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    # 创建流式处理器
    streamer = TextStreamer(
        tokenizer,
        # skip_prompt=True,  # 跳过原始提示
        # timeout=0.1,  # 输出间隔A
        # skip_special_tokens=True
    )

    # 开始流式生成
    model.generate(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )


# 使用示例
torch.cuda.empty_cache()
stream_completion("你到底是谁呢？")