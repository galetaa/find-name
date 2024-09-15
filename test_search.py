import requests
import time
import json
from tqdm import tqdm  # Для прогресс-бара
from sklearn.metrics import precision_score, recall_score, f1_score

API_URL = "https://caila.io/api/adapters/openai/chat/completions"
API_KEY = "1000169780.107212.mRxJYRc5QJQr3PsxRMAT2xY5ZdiukBrGaBuKcFB0"

models = [
    # "just-ai/claude/",
    "just-ai/openai-proxy/gpt-4o",
    "just-ai/vllm-qwen2-72b-awq",
    "just-ai/vllm-llama3.1-70b-4q",
    "just-ai/gemini"
]


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def call_caila_model(text: str, model_id: str):
    """Запрос к модели"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [{"role": "user",
                      "content": f"Выдели все имена, фамилии и отчества, которые встречаются в тексте, "
                                 f"и верни их в точности в той форме, в которой они написаны в тексте. "
                                 f"Не изменяй регистр, склонение или другие свойства слов. Если в тексте "
                                 f"есть хотя бы одно имя, фамилия или отчество, верни их. Если таких "
                                 f"компонентов нет, верни пустую строку. Не добавляй никакого другого "
                                 f"текста, кроме найденных имен, фамилий или отчеств.: {text}"}],
        "stream": False
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def process_llm_response(llm_response):
    found_entities = set()

    for choice in llm_response.get("choices", []):
        entity_text = choice.get("message", {}).get("content", "").strip()

        if entity_text:
            entity_words = entity_text.split()
            found_entities.update(entity_words)

    return found_entities


def extract_true_entities(sentence_tokens, true_entities_indices):
    """Получение верных слов по индексам"""
    true_entities = set()

    for entity in true_entities_indices:
        entity_words = [sentence_tokens[i] for i in entity]
        true_entities.update(entity_words)

    return true_entities


def calculate_metrics(true_entities, predicted_entities):
    """Расчет precision, recall, f1-score"""

    if not true_entities and not predicted_entities:  # Если оба множества пустые, считаем метрики равными 1
        return 1.0, 1.0, 1.0

    tp = len(true_entities & predicted_entities)

    precision = tp / len(predicted_entities) if predicted_entities else 0
    recall = tp / len(true_entities) if true_entities else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

    return precision, recall, f1


def test_model_on_dataset(model_id: str, dataset):
    """Тестирование модели на N примерах"""
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_chars = 0
    total_time = 0

    for example in tqdm(dataset, desc=f"Тестирование {model_id}"):
        sentence_tokens = example['sentence']
        text = " ".join(sentence_tokens)
        true_entities_indices = [e['index'] for e in example['ner'] if e['type'] == 'PER']
        true_entities = extract_true_entities(sentence_tokens, true_entities_indices)

        start_time = time.time()

        llm_response = call_caila_model(text, model_id)
        predicted_entities = process_llm_response(llm_response)
        precision, recall, f1 = calculate_metrics(true_entities, predicted_entities)

        elapsed_time = time.time() - start_time

        total_chars += len(" ".join(sentence_tokens))
        total_time += elapsed_time

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_precision = total_precision / len(dataset)
    avg_recall = total_recall / len(dataset)
    avg_f1 = total_f1 / len(dataset)

    speed = total_chars / total_time

    return avg_precision, avg_recall, avg_f1, speed


def main():
    data_path = 'data/nerus_10K.json'  # Путь к вашему сгенерированному датасету
    n = 100  # Количество примеров для тестирования

    data = load_json(data_path)[:n]

    for model in models:
        avg_precision, avg_recall, avg_f1, speed = test_model_on_dataset(model, data)

        print(f"Модель {model}:")
        print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        print(f"Скорость: {speed:.2f} симв/сек")


if __name__ == "__main__":
    main()
