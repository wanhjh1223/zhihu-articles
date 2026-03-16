"""
Gradio Web UI 部署
"""

import argparse
import gradio as gr
from ...llm_training.models.base_model import LLMModel


class ChatInterface:
    """聊天界面"""
    
    def __init__(self, model_path: str):
        print(f"正在加载模型: {model_path}")
        self.model = LLMModel(
            model_name_or_path=model_path,
            load_in_4bit=True,
        )
        print("模型加载完成")
    
    def generate(self, message: str, history: list, 
                 max_tokens: int, temperature: float) -> str:
        """生成回复"""
        # 构建对话历史
        prompt = ""
        for human, assistant in history:
            prompt += f"<|user|>\n{human}\n<|assistant|>\n{assistant}\n"
        prompt += f"<|user|>\n{message}\n<|assistant|>\n"
        
        # 生成
        response = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response
    
    def create_ui(self) -> gr.Blocks:
        """创建 UI"""
        with gr.Blocks(title="LLM Chat") as demo:
            gr.Markdown("# 🤖 LLM 对话系统")
            
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="输入消息")
            
            with gr.Row():
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="最大长度")
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="温度")
            
            btn = gr.Button("发送")
            clear = gr.Button("清空")
            
            def respond(message, chat_history, max_t, temp):
                bot_message = self.generate(message, chat_history, max_t, temp)
                chat_history.append((message, bot_message))
                return "", chat_history
            
            btn.click(respond, [msg, chatbot, max_tokens, temperature], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        return demo


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Gradio Web UI")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    
    args = parser.parse_args()
    
    interface = ChatInterface(args.model)
    demo = interface.create_ui()
    
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
