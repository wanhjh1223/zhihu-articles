"""
部署示例
演示如何部署模型为 API 服务
"""

import sys
sys.path.insert(0, '../../src')


def example_api_deployment():
    """示例：API 部署"""
    print("=" * 50)
    print("示例：API 部署")
    print("=" * 50)
    
    print("""
启动 API 服务:

```bash
# 使用命令行
python -m src.common.deployment.api_server \\
    --model ./outputs/llm_sft/final \\
    --host 0.0.0.0 \\
    --port 8000

# 或使用脚本
bash scripts/start_api_server.sh
```

API 使用示例:

```python
import requests

# 调用聊天接口
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [
        {"role": "user", "content": "你好！"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
})

print(response.json())
```
""")


def example_gradio_ui():
    """示例：Gradio UI"""
    print("\n" + "=" * 50)
    print("示例：Gradio UI")
    print("=" * 50)
    
    print("""
启动 Gradio 界面:

```bash
# 使用命令行
python -m src.common.deployment.gradio_ui \\
    --model ./outputs/llm_sft/final \\
    --port 7860
```

然后在浏览器打开 http://localhost:7860
""")


def example_inference():
    """示例：推理代码"""
    print("\n" + "=" * 50)
    print("示例：推理代码")
    print("=" * 50)
    
    code = '''
from src.llm_training.models.base_model import load_model_for_inference

# 加载模型
model = load_model_for_inference(
    model_path="./outputs/llm_sft/final",
    load_in_4bit=True
)

# 生成回复
response = model.generate(
    prompt="介绍一下深度学习：",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

print(response)
'''
    print(code)


if __name__ == "__main__":
    example_api_deployment()
    example_gradio_ui()
    example_inference()
    
    print("\n" + "=" * 50)
    print("示例代码查看完成！")
    print("=" * 50)
