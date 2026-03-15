"""
模型评估模块
支持多种评估指标和基准测试
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """评估配置"""
    # 模型配置
    model_path: str
    
    # 评估数据
    eval_data_path: Optional[str] = None
    eval_tasks: List[str] = None  # ['perplexity', 'generation', 'mcqa']
    
    # 生成配置
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # 评估配置
    batch_size: int = 4
    num_samples: Optional[int] = None  # 评估样本数，None表示全部
    
    # 输出配置
    output_dir: str = "./eval_results"
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.eval_tasks is None:
            self.eval_tasks = ['perplexity']


class PerplexityEvaluator:
    """困惑度评估器"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def evaluate(self, texts: List[str], max_length: int = 2048) -> Dict[str, float]:
        """
        计算困惑度
        
        PPL = exp(平均交叉熵损失)
        越低越好
        """
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            # Tokenize
            encodings = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors='pt',
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            
            # 计算loss
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 计算有效token数
            num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
        }


class GenerationEvaluator:
    """生成质量评估器"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """生成回复"""
        generations = []
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                max_length=self.tokenizer.model_max_length - max_new_tokens,
                truncation=True,
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 解码生成的部分
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            generations.append(generated_text)
        
        return generations
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算BLEU分数
        
        简化版本：使用n-gram重叠计算
        """
        def get_ngrams(text, n):
            tokens = text.lower().split()
            return set(zip(*[tokens[i:] for i in range(n)]))
        
        def compute_bleu(pred, ref, max_n=4):
            scores = []
            for n in range(1, max_n + 1):
                pred_ngrams = get_ngrams(pred, n)
                ref_ngrams = get_ngrams(ref, n)
                
                if len(pred_ngrams) == 0:
                    scores.append(0.0)
                    continue
                
                matches = len(pred_ngrams & ref_ngrams)
                precision = matches / len(pred_ngrams)
                scores.append(precision)
            
            # 几何平均
            import math
            if all(s > 0 for s in scores):
                return math.exp(sum(math.log(s) for s in scores) / len(scores))
            return 0.0
        
        bleu_scores = [
            compute_bleu(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        
        return {
            'bleu': sum(bleu_scores) / len(bleu_scores),
            'bleu_scores': bleu_scores,
        }
    
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        简化版本：计算unigram和bigram召回率
        """
        def get_unigrams(text):
            return set(text.lower().split())
        
        def get_bigrams(text):
            tokens = text.lower().split()
            return set(zip(tokens[:-1], tokens[1:]))
        
        rouge1_scores = []
        rouge2_scores = []
        
        for pred, ref in zip(predictions, references):
            # ROUGE-1
            pred_unigrams = get_unigrams(pred)
            ref_unigrams = get_unigrams(ref)
            if len(ref_unigrams) > 0:
                rouge1 = len(pred_unigrams & ref_unigrams) / len(ref_unigrams)
                rouge1_scores.append(rouge1)
            
            # ROUGE-2
            pred_bigrams = get_bigrams(pred)
            ref_bigrams = get_bigrams(ref)
            if len(ref_bigrams) > 0:
                rouge2 = len(pred_bigrams & ref_bigrams) / len(ref_bigrams)
                rouge2_scores.append(rouge2)
        
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        }


class MCQAEvaluator:
    """多项选择题评估器"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def evaluate_choice(
        self,
        question: str,
        choices: List[str],
        answer_index: int,
    ) -> Tuple[int, float]:
        """
        评估单选题
        
        返回：(预测答案索引, 置信度)
        """
        choice_probs = []
        
        for choice in choices:
            # 构建完整文本
            full_text = f"{question} {choice}"
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors='pt',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).to(self.device)
            
            # 计算loss（作为置信度的代理）
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            
            # 转换为概率（loss越低，概率越高）
            prob = torch.exp(-torch.tensor(loss)).item()
            choice_probs.append(prob)
        
        # 选择概率最高的
        pred_index = int(torch.tensor(choice_probs).argmax())
        confidence = choice_probs[pred_index] / sum(choice_probs)
        
        return pred_index, confidence
    
    def evaluate_batch(
        self,
        questions: List[str],
        all_choices: List[List[str]],
        answer_indices: List[int],
    ) -> Dict[str, float]:
        """批量评估"""
        correct = 0
        total = len(questions)
        confidences = []
        
        for q, choices, ans_idx in zip(questions, all_choices, answer_indices):
            pred_idx, confidence = self.evaluate_choice(q, choices, ans_idx)
            if pred_idx == ans_idx:
                correct += 1
            confidences.append(confidence)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_confidence': sum(confidences) / len(confidences),
        }


class Evaluator:
    """主评估器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        self._setup_model()
        
        # 初始化子评估器
        self.ppl_evaluator = PerplexityEvaluator(
            self.model, self.tokenizer, self.device
        )
        self.gen_evaluator = GenerationEvaluator(
            self.model, self.tokenizer, self.device
        )
        self.mcqa_evaluator = MCQAEvaluator(
            self.model, self.tokenizer, self.device
        )
        
    def _setup_model(self):
        """加载模型"""
        logger.info(f"加载模型: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=True,
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        logger.info(f"模型加载完成")
        
    def evaluate_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """评估困惑度"""
        logger.info("评估困惑度...")
        results = self.ppl_evaluator.evaluate(texts, self.config.max_length)
        logger.info(f"PPL: {results['perplexity']:.2f}")
        return results
    
    def evaluate_generation(
        self,
        prompts: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """评估生成质量"""
        logger.info("评估生成质量...")
        
        # 生成
        predictions = self.gen_evaluator.generate(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
        )
        
        # 计算指标
        bleu_results = self.gen_evaluator.evaluate_bleu(predictions, references)
        rouge_results = self.gen_evaluator.evaluate_rouge(predictions, references)
        
        results = {
            **bleu_results,
            **rouge_results,
            'predictions': predictions if self.config.save_predictions else None,
        }
        
        logger.info(f"BLEU: {results['bleu']:.4f}, ROUGE-1: {results['rouge1']:.4f}")
        return results
    
    def evaluate_mcqa(
        self,
        questions: List[str],
        choices: List[List[str]],
        answer_indices: List[int],
    ) -> Dict[str, float]:
        """评估多项选择"""
        logger.info("评估MCQA...")
        results = self.mcqa_evaluator.evaluate_batch(
            questions, choices, answer_indices
        )
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        return results
    
    def run_full_eval(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整评估"""
        logger.info("开始完整评估...")
        
        all_results = {
            'model_path': self.config.model_path,
            'eval_tasks': self.config.eval_tasks,
        }
        
        # 困惑度评估
        if 'perplexity' in self.config.eval_tasks and 'texts' in eval_data:
            all_results['perplexity'] = self.evaluate_perplexity(eval_data['texts'])
        
        # 生成评估
        if 'generation' in self.config.eval_tasks:
            if 'prompts' in eval_data and 'references' in eval_data:
                all_results['generation'] = self.evaluate_generation(
                    eval_data['prompts'],
                    eval_data['references'],
                )
        
        # MCQA评估
        if 'mcqa' in self.config.eval_tasks:
            if all(k in eval_data for k in ['questions', 'choices', 'answer_indices']):
                all_results['mcqa'] = self.evaluate_mcqa(
                    eval_data['questions'],
                    eval_data['choices'],
                    eval_data['answer_indices'],
                )
        
        # 保存结果
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "eval_results.json"
        
        # 移除不可序列化的数据
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        clean_results = clean_for_json(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {output_file}")


def main():
    """测试评估器"""
    config = EvalConfig(
        model_path="gpt2",
        eval_tasks=['perplexity', 'generation'],
        output_dir="./eval_results/test",
    )
    
    evaluator = Evaluator(config)
    
    # 测试数据
    eval_data = {
        'texts': [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的核心技术。",
        ],
        'prompts': [
            "人工智能是",
            "机器学习是",
        ],
        'references': [
            "人工智能是计算机科学的重要分支，研究如何让机器模拟人类智能。",
            "机器学习是人工智能的一种方法，使计算机能够从数据中学习。",
        ],
    }
    
    results = evaluator.run_full_eval(eval_data)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
