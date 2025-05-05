from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  
    base_url="http://0.0.0.0:8947/v1"
)

print(client.models.list())

stream = client.chat.completions.create(
    model="Qwen2.5-1.5B-Instruct-GPTQ-Int4-2",
    messages=[
        {"role": "user", "content": "你好，請介紹一下你是誰？"},
    ],
    temperature=0.7,
    stream=True  
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
