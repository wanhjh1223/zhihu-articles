"""
模型评估工具
支持 LLM 和 VLM 的自动评估
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from ...llm_training.models.base_model import LLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """评估配置"""
    model_path: str
    eval_data_path: str
    output_path: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1


class LLMEvaluator:
    """LLM 评估器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        
        logger.info(f"正在加载模型: {config.model_path}")
        self.model = LLMModel(
            model_name_or_path=config.model_path,
            load_in_4bit=True,
        )
        self.tokenizer = self.model.tokenizer
    
    def evaluate(self) -> Dict:
        """执行评估"""
        logger.info(f"正在加载评估数据: {self.config.eval_data_path}")
        
        eval_data = self._load_eval_data()
        results = []
        
        logger.info(f"开始评估，共 {len(eval_data)} 条数据...")
        
        for item in tqdm(eval_data):
            prompt = item.get('prompt', item.get('instruction', ''))
            reference = item.get('reference', item.get('output', ''))
            
            # 生成回复
            generated = self.model.generate(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            results.append({
                'prompt': prompt,
                'reference': reference,
                'generated': generated,
            })
        
        # 计算指标
        metrics = self._compute_metrics(results)
        
        # 保存结果
        self._save_results(results, metrics)
        
        return metrics
    
    def _load_eval_data(self) -> List[Dict]:
        """加载评估数据"""
        data = []
        with open(self.config.eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    
    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """计算评估指标"""
        from rouge import Rouge
        
        rouge = Rouge()
        references = [r['reference'] for r in results]
        generated = [r['generated'] for r in results]
        
        # ROUGE 分数
        try:
            rouge_scores = rouge.get_scores(generated, references, avg=True)
        except:
            rouge_scores = {}
        
        # 平均长度
        avg_ref_len = np.mean([len(r) for r in references])
        avg_gen_len = np.mean([len(g) for g in generated])
        
        metrics = {
            'rouge-1': rouge_scores.get('rouge-1', {}).get('f', 0),
            'rouge-2': rouge_scores.get('rouge-2', {}).get('f', 0),
            'rouge-l': rouge_scores.get('rouge-l', {}).get('f', 0),
            'avg_reference_length': avg_ref_len,
            'avg_generated_length': avg_gen_len,
            'num_samples': len(results),
        }
        
        return metrics
    
    def _save_results(self, results: List[Dict], metrics: Dict):
        """保存结果"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 保存指标
        metrics_path = output_path.with_suffix('.metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {output_path}")
        logger.info(f"评估指标: {metrics}")


class VLMEvaluator:
    """VLM 评估器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        # TODO: 加载 VLM 模型
    
    def evaluate(self) -> Dict:
        """执行评估"""
        logger.info("VLM 评估功能待实现")
        return {}


class BenchmarkEvaluator:
    """基准测试评估器"""
    
    BENCHMARKS = {
        'ceval': 'C-Eval 中文知识评估',
        'cmmlu': 'CMMLU 中文多任务',
        'mmlu': 'MMLU 英文知识评估',
        'gsm8k': 'GSM8K 数学推理',
        'human_eval': 'HumanEval 代码生成',
    }
    
    def __init__(self, model_path: str, benchmark: str):
        self.model_path = model_path
        self.benchmark = benchmark
    
    def run(self) -> Dict:
        """运行基准测试"""
        logger.info(f"运行基准测试: {self.BENCHMARKS.get(self.benchmark, self.benchmark)}")
        
        # 这里可以集成 lm-evaluation-harness 或其他评估框架
        # 简化实现
        
        return {
            'benchmark': self.benchmark,
            'score': 0.0,
            'note': '请使用 lm-evaluation-harness 进行完整评估',
        }


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--data", required=True, help="评估数据路径")
    parser.add_argument("--output", required=True, help="输出路径")
    parser.add_argument("--type", choices=['llm', 'vlm'], default='llm',
                       help="模型类型")
    parser.add_argument("--benchmark", help="基准测试名称")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_path=args.model,
        eval_data_path=args.data,
        output_path=args.output,
    )
    
    if args.benchmark:
        evaluator = BenchmarkEvaluator(args.model, args.benchmark)
        metrics = evaluator.run()
        print(f"基准测试结果: {metrics}")
    elif args.type == 'llm':
        evaluator = LLMEvaluator(config)
        metrics = evaluator.evaluate()
        print(f"评估指标: {metrics}")
    else:
        evaluator = VLMEvaluator(config)
        metrics = evaluator.evaluate()


if __name__ == "__main__":
    main()
