"""
模型错误类型分析脚本。

该脚本会加载已训练好的 BiLSTM+CRF 模型，对测试集进行预测，
输出以下信息：
1. 基本的评估指标（精确率、召回率、F1 等）。
2. 预测错误的类型统计（gold->pred 计数）。
3. 每种错误类型的典型错误案例，逐字展示正确标签与预测标签。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

from data import build_corpus
from evaluating import Metrics
from utils import extend_maps, load_model, prepocess_data_for_lstmcrf

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ckpts" / "bilstm_crf.pkl"


def _ensure_best_model(model) -> None:
    """保证反序列化后的模型可以直接用于预测。"""
    if model.best_model is None:
        model.best_model = deepcopy(model.model)
    # 去除 PackedSequence 的警告，兼容 GPU/CPU
    if hasattr(model.best_model, "bilstm"):
        bilstm_block = getattr(model.best_model, "bilstm")
        if hasattr(bilstm_block, "bilstm"):
            bilstm_block.bilstm.flatten_parameters()


def _collect_errors(
    sentences: Sequence[Sequence[str]],
    golden_tags: Sequence[Sequence[str]],
    pred_tags: Sequence[Sequence[str]],
    max_examples_per_type: int = 3,
) -> Tuple[Counter, DefaultDict[str, List[Dict[str, str]]]]:
    """统计错误类型并返回典型案例。"""
    error_counter: Counter = Counter()
    error_examples: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)

    for words, gold_seq, pred_seq in zip(sentences, golden_tags, pred_tags):
        gold_entities = _extract_entities(words, gold_seq)
        pred_entities = _extract_entities(words, pred_seq)

        for idx, (gold, pred) in enumerate(zip(gold_seq, pred_seq)):
            if gold != pred:
                error_type = f"{gold}->{pred}"
                error_counter[error_type] += 1
                if len(error_examples[error_type]) < max_examples_per_type:
                    error_examples[error_type].append(
                        {
                            "sentence": "".join(words),
                            "words": words,
                            "gold_tags": gold_seq,
                            "pred_tags": pred_seq,
                            "gold_entities": gold_entities,
                            "pred_entities": pred_entities,
                        }
                    )
    return error_counter, error_examples


def _print_error_report(error_counter: Counter, error_examples: Dict[str, List[Dict[str, str]]]) -> None:
    print("\n错误类型统计（gold->pred）:")
    for err_type, count in error_counter.most_common():
        print(f"  {err_type}: {count}")

    print("\n典型错误案例:")
    for err_type, examples in error_examples.items():
        print(f"\n[{err_type}] 展示 {len(examples)} 个案例：")
        for case in examples:
            gold_entities = _format_entities(case["gold_entities"])
            pred_entities = _format_entities(case["pred_entities"])

            print(f"句子          : {case['sentence']}")

            words = case["words"]
            gold_tags = case["gold_tags"]
            pred_tags = case["pred_tags"]

            # 逐字打印标签
            print("字/词         :", " ".join(words))
            print("Gold 标签     :", " ".join(gold_tags))
            print("Pred 标签     :", " ".join(pred_tags))



def _extract_entities(words: Sequence[str], tags: Sequence[str]) -> List[Dict[str, str]]:
    """将 BIO 标注转换为实体列表。"""
    entities: List[Dict[str, str]] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type = tag[2:]
            start = i
            end = i + 1
            while end < len(tags) and tags[end].startswith("I-") and tags[end][2:] == entity_type:
                end += 1
            entity_text = "".join(words[start:end])
            entities.append({"type": entity_type, "text": entity_text, "span": f"{start}-{end}"})
            i = end
        else:
            i += 1
    return entities


def _format_entities(entities: Sequence[Dict[str, str]]) -> str:
    if not entities:
        return "(无)"
    return " | ".join(f"{e['type']}:{e['text']}[{e['span']}]" for e in entities)


def evaluate(model_path: Path, remove_o: bool = False) -> None:
    print("读取数据...")
    _, _, word2id, tag2id = build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 保留原始句子，便于展示错误案例
    display_word_lists = [list(seq) for seq in test_word_lists]

    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )

    print(f"加载模型: {model_path}")
    bilstm_model = load_model(str(model_path))
    _ensure_best_model(bilstm_model)

    print("开始预测...")
    pred_tag_lists, target_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, crf_word2id, crf_tag2id
    )

    print("\n基础评估指标：")
    metrics = Metrics(target_tag_lists, pred_tag_lists, remove_O=remove_o)
    metrics.report_scores()

    error_counter, error_examples = _collect_errors(
        display_word_lists, target_tag_lists, pred_tag_lists
    )
    _print_error_report(error_counter, error_examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="模型预测错误类型分析工具")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="模型文件路径，默认为 ckpts/bilstm_crf.pkl",
    )
    parser.add_argument(
        "--remove_o",
        action="store_true",
        help="评估与统计时是否移除 O 标签",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {args.model_path}")

    evaluate(args.model_path, remove_o=args.remove_o)


if __name__ == "__main__":
    main()
