from collections import defaultdict
from razdel import tokenize
import json


def parse_conllu_file(filepath, n=100):
    sents = defaultdict(list)
    temp = []
    curr_sent = ''
    sentences_count = 0
    list_for_json = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if '# text = ' in line:
                if temp and curr_sent:
                    sents[curr_sent].append(temp)
                    temp = []
                    json_data = process_sentence(sents, curr_sent)
                    if json_data:
                        list_for_json.append(json_data)
                        sentences_count += 1
                    if sentences_count >= n:
                        break
                curr_sent = line[9:].strip()

            splits = line.split('\t')
            if len(splits) > 1:
                if splits[-1].strip() in ['Tag=I-ORG', 'Tag=I-PER', 'Tag=I-LOC']:
                    temp.append((splits[1], splits[-1].strip()))
                elif splits[-1].strip() in ['Tag=B-ORG', 'Tag=B-PER', 'Tag=B-LOC']:
                    if not temp:
                        temp.append((splits[1], splits[-1].strip()))
                    else:
                        sents[curr_sent].append(temp)
                        temp = [(splits[1], splits[-1].strip())]

    return list_for_json


def process_sentence(sents, curr_sent):
    temp = {'sentence': list(map(lambda x: x.text, tokenize(curr_sent)))}

    if len(temp['sentence']) > 100:
        return None

    temp_ner = []
    for ner in sents[curr_sent]:
        d = {}
        ixs = []
        type_ = ''
        for NER, typ in ner:
            ixs.append(temp['sentence'].index(NER))
            type_ = typ[6:9]
        d['index'] = ixs
        d['type'] = type_
        temp_ner.append(d)
    temp['ner'] = temp_ner
    return temp


def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def main(input_path, output_path, max_sentences):
    parsed_data = parse_conllu_file(input_path, max_sentences)
    save_to_json(parsed_data, output_path)
    print(f"Датасет сохранен в {output_path}, количество предложений: {len(parsed_data)}")


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/nerus_lenta.conllu'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/parsed_data.json'
    max_sentences = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    main(input_file, output_file, max_sentences)
