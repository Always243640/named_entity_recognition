import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, List, Tuple

from data import build_corpus
from evaluate import bilstm_train_and_eval, crf_train_eval, ensemble_evaluate, hmm_train_eval
from evaluating import Metrics
from utils import extend_maps, load_model, prepocess_data_for_lstmcrf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DataNER"
HMM_MODEL_PATH = PROJECT_ROOT / "ckpts" / "hmm.pkl"
CRF_MODEL_PATH = PROJECT_ROOT / "ckpts" / "crf.pkl"
BiLSTM_MODEL_PATH = PROJECT_ROOT / "ckpts" / "bilstm.pkl"
BiLSTMCRF_MODEL_PATH = PROJECT_ROOT / "ckpts" / "bilstm_crf.pkl"

ModelProgressCallback = Callable[[str, int], None]


MODEL_OPTIONS = [
    "HMM",
    "CRF",
    "BiLSTM",
    "BiLSTM-CRF",
]


class UnknownModelError(ValueError):
    """Raised when the selected model is not supported."""


def _notify(callback: ModelProgressCallback, message: str, progress: int) -> None:
    if callback:
        callback(message, progress)


def _prepare_sentences(text: str) -> List[List[str]]:
    sentences = [list(line.strip()) for line in text.splitlines() if line.strip()]
    if not sentences:
        raise ValueError("请输入需要预测的文本内容")
    return sentences


def _extract_entities(word_list: List[str], tag_list: List[str]) -> List[str]:
    entities = []
    current_tokens: List[str] = []
    current_type: str = None

    def _close_entity():
        if current_tokens and current_type:
            entities.append(f"{current_type}: {''.join(current_tokens)}")

    for word, tag in zip(word_list, tag_list):
        if tag == "O" or tag is None:
            _close_entity()
            current_tokens, current_type = [], None
            continue

        try:
            prefix, ent_type = tag.split("-", 1)
        except ValueError:
            _close_entity()
            current_tokens, current_type = [], None
            continue

        if prefix in {"B", "S"}:
            _close_entity()
            current_tokens = [word]
            current_type = ent_type
            if prefix == "S":
                _close_entity()
                current_tokens, current_type = [], None
        elif prefix in {"I", "M"}:
            if current_type == ent_type:
                current_tokens.append(word)
            else:
                current_tokens, current_type = [word], ent_type
        elif prefix == "E":
            if current_type != ent_type:
                current_tokens, current_type = [word], ent_type
            else:
                current_tokens.append(word)
            _close_entity()
            current_tokens, current_type = [], None
        else:
            _close_entity()
            current_tokens, current_type = [], None

    _close_entity()
    return entities


def _format_predictions(sentences: List[List[str]], tag_lists: List[List[str]]) -> str:
    lines: List[str] = []
    for idx, (words, tags) in enumerate(zip(sentences, tag_lists), start=1):
        labeled = " ".join(f"{w}/{t}" for w, t in zip(words, tags))
        entities = _extract_entities(words, tags)
        entity_line = "，".join(entities) if entities else "未识别到实体"
        lines.append(f"句子{idx}：{''.join(words)}")
        lines.append(f"标注序列：{labeled}")
        lines.append(f"实体：{entity_line}")
    return "\n".join(lines)


def _load_datasets():
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=str(DATA_DIR))
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir=str(DATA_DIR))
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=str(DATA_DIR))
    return (
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id,
    )


def _capture_stdout(func, *args, **kwargs) -> Tuple[str, object]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return buffer.getvalue(), result


def train_selected_model(model_name: str, callback: ModelProgressCallback = None) -> str:
    remove_O = False
    _notify(callback, None, 5)
    train_data, dev_data, test_data, word2id, tag2id = _load_datasets()

    def log_and_notify(message: str, progress: int):
        _notify(callback, message, progress)

    log_and_notify(f"正在训练{model_name}模型...", 20)
    if model_name == "HMM":
        stdout, _ = _capture_stdout(
            hmm_train_eval, train_data, test_data, word2id, tag2id, remove_O=remove_O
        )
    elif model_name == "CRF":
        stdout, _ = _capture_stdout(
            crf_train_eval, train_data, test_data, remove_O=remove_O
        )
    elif model_name == "BiLSTM":
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        stdout, _ = _capture_stdout(
            bilstm_train_and_eval,
            train_data,
            dev_data,
            test_data,
            bilstm_word2id,
            bilstm_tag2id,
            False,
            remove_O,
        )
    elif model_name == "BiLSTM-CRF":
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        processed_train = prepocess_data_for_lstmcrf(*train_data)
        processed_dev = prepocess_data_for_lstmcrf(*dev_data)
        processed_test = prepocess_data_for_lstmcrf(*test_data, test=True)
        stdout, _ = _capture_stdout(
            bilstm_train_and_eval,
            processed_train,
            processed_dev,
            processed_test,
            crf_word2id,
            crf_tag2id,
            True,
            remove_O,
        )
    else:
        raise UnknownModelError(f"不支持的模型类型：{model_name}")

    _notify(callback, stdout.strip(), 90)
    _notify(callback, None, 100)
    return stdout


def evaluate_selected_model(model_name: str, callback: ModelProgressCallback = None) -> str:
    _notify(callback, None, 5)
    train_data, _, test_data, word2id, tag2id = _load_datasets()
    test_word_lists, test_tag_lists = test_data
    remove_O = False

    def report(message: str, progress: int):
        _notify(callback, message, progress)

    report(f"加载{model_name}模型并开始评估...", 15)
    if model_name == "HMM":
        hmm_model = load_model(str(HMM_MODEL_PATH))
        stdout, pred_tag_lists = _capture_stdout(
            hmm_model.test, test_word_lists, word2id, tag2id
        )
    elif model_name == "CRF":
        crf_model = load_model(str(CRF_MODEL_PATH))
        stdout, pred_tag_lists = _capture_stdout(crf_model.test, test_word_lists)
    elif model_name == "BiLSTM":
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        bilstm_model = load_model(str(BiLSTM_MODEL_PATH))
        bilstm_model.model.bilstm.flatten_parameters()
        stdout, result = _capture_stdout(
            bilstm_model.test,
            test_word_lists,
            test_tag_lists,
            bilstm_word2id,
            bilstm_tag2id,
        )
        pred_tag_lists, test_tag_lists = result
    elif model_name == "BiLSTM-CRF":
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        bilstm_model = load_model(str(BiLSTMCRF_MODEL_PATH))
        bilstm_model.model.bilstm.bilstm.flatten_parameters()
        processed_test_word_lists, processed_test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
        stdout, result = _capture_stdout(
            bilstm_model.test,
            processed_test_word_lists,
            processed_test_tag_lists,
            crf_word2id,
            crf_tag2id,
        )
        pred_tag_lists, test_tag_lists = result
    else:
        raise UnknownModelError(f"不支持的模型类型：{model_name}")

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metric_buffer, _ = _capture_stdout(metrics.report_scores)
    confusion_buffer, _ = _capture_stdout(metrics.report_confusion_matrix)
    ensemble_stdout = ""

    report(stdout.strip(), 60)
    report(metric_buffer.strip(), 80)
    report(confusion_buffer.strip(), 90)

    if model_name != "HMM":
        ensemble_stdout, _ = _capture_stdout(
            ensemble_evaluate, [pred_tag_lists], test_tag_lists
        )
        report(ensemble_stdout.strip(), 95)

    report(None, 100)
    return "\n".join(filter(None, [stdout, metric_buffer, confusion_buffer, ensemble_stdout]))


def predict_entities(model_name: str, text: str, callback: ModelProgressCallback = None) -> str:
    sentences = _prepare_sentences(text)
    _notify(callback, None, 5)
    _, _, _, word2id, tag2id = _load_datasets()

    _notify(callback, f"加载{model_name}模型...", 15)

    if model_name == "HMM":
        hmm_model = load_model(str(HMM_MODEL_PATH))
        pred_tag_lists = hmm_model.test(sentences, word2id, tag2id)
    elif model_name == "CRF":
        crf_model = load_model(str(CRF_MODEL_PATH))
        pred_tag_lists = crf_model.test(sentences)
    elif model_name == "BiLSTM":
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        bilstm_model = load_model(str(BiLSTM_MODEL_PATH))
        bilstm_model.model.bilstm.flatten_parameters()
        pred_tag_lists, _ = bilstm_model.test(
            sentences, None, bilstm_word2id, bilstm_tag2id
        )
    elif model_name == "BiLSTM-CRF":
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        processed_sentences = [list(words) for words in sentences]
        dummy_tags = [[] for _ in processed_sentences]
        processed_sentences, dummy_tags = prepocess_data_for_lstmcrf(
            processed_sentences, dummy_tags, test=True
        )
        bilstm_model = load_model(str(BiLSTMCRF_MODEL_PATH))
        bilstm_model.model.bilstm.bilstm.flatten_parameters()
        pred_tag_lists, _ = bilstm_model.test(
            processed_sentences, dummy_tags, crf_word2id, crf_tag2id
        )
    else:
        raise UnknownModelError(f"不支持的模型类型：{model_name}")

    _notify(callback, "预测完成，正在整理结果...", 80)
    formatted = _format_predictions(sentences, pred_tag_lists)
    _notify(callback, formatted, 100)
    return formatted
