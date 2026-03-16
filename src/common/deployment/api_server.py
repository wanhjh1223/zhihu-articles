"""
API 服务部署
支持 LLM 和 VLM 的推理服务
"""

import os
import json
import logging
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch

from ...llm_training.models.base_model import LLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型实例
model_instance = None


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    usage: Dict


class GenerateRequest(BaseModel):
    """生成请求"""
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class VLMRequest(BaseModel):
    """VLM 请求"""
    image: str  # base64 编码
    prompt: str
    max_new_tokens: int = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("服务启动中...")
    yield
    logger.info("服务关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="LLM-VLM API Server",
    description="大语言模型和视觉语言模型推理服务",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LLM-VLM API Server",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model_instance is not None}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 风格聊天接口"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 格式化消息为 prompt
        prompt = format_messages(request.messages)
        
        # 生成回复
        response_text = model_instance.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(response_text),
                "total_tokens": len(prompt) + len(response_text),
            }
        }
        
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    """文本生成接口"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        response_text = model_instance.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(response_text),
            }
        )
        
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/vlm/chat")
async def vlm_chat(request: VLMRequest):
    """VLM 多模态聊天接口"""
    # TODO: 实现 VLM 推理
    return {"message": "VLM 推理功能开发中"}


def format_messages(messages: List[Dict[str, str]]) -> str:
    """格式化消息列表为 prompt"""
    prompt = ""
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            prompt += f"<|system|>\n{content}\n"
        elif role == 'user':
            prompt += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    return prompt


def load_model(model_path: str):
    """加载模型"""
    global model_instance
    
    logger.info(f"正在加载模型: {model_path}")
    model_instance = LLMModel(
        model_name_or_path=model_path,
        load_in_4bit=True,
        device_map="auto",
    )
    logger.info("模型加载完成")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API 服务")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model)
    
    # 启动服务
    uvicorn.run(
        "src.common.deployment.api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
