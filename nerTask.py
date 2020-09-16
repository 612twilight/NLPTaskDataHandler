import os
import json
from collections import OrderedDict
from tokenizers import BertWordPieceTokenizer


def convert_to_conll(data, filename, data_dir=".", suffix=""):
    """
    data: list of a labeled data
    [
    {"text":xxxxx,"labels":{"label1":[[0,2,'ORG'],[11.17.'PER]],"label2":[[0,2,'ORG'],[11.17.'PER]]}}
    ]
    其中表示位置的数值是左闭右开比如[0,2,'ORG']表示[0,2]之间的下标是ORG的字符
    :return:
    """
    file_path = os.path.join(data_dir, filename + suffix)
    with open(file_path, 'w', encoding='utf8') as writer:
        for item in data:
            text = item['text']
            labels = item['labels']
            BIO_labels = OrderedDict()
            for label_key in labels:
                BIO_label = ["O"] * len(text)
                anno_labels = list(sorted(labels[label_key], key=lambda x: x[0]))
                if not _pass_conflict_check(anno_labels):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise Exception("标注冲突，数据为：" + json.dumps(item, ensure_ascii=False))
                else:
                    for single_label in anno_labels:
                        for index in range(single_label[0], single_label[1]):
                            if index == single_label[0]:
                                BIO_label[index] = "B-" + single_label[2]
                            else:
                                BIO_label[index] = "O-" + single_label[2]
                    BIO_labels[label_key] = BIO_label
            labels = [i[1] for i in BIO_labels.items()]
            for i in range(len(text)):
                tmp = [lab[i] for lab in labels]
                writer.write(text[i] + "\t" + "\t".join(tmp) + "\n")
            writer.write("\n")


def _pass_conflict_check(anno_labels):
    """
    已经过排序
    :param anno_labels:
    :return:
    """
    for i in range(len(anno_labels) - 1):
        if anno_labels[i][1] > anno_labels[i + 1][0]:
            return False
    return True


def convert_to_conll_with_chinese_bert(data, filename, data_dir=".", suffix=""):
    tokenizer = BertWordPieceTokenizer("./data/bert-chinese-vocab.txt")
    clean_data = []
    for item in data:
        text = item['text']
        labels = item['labels']
        positions = set()
        BIO_labels = OrderedDict()
        for label_key in labels:
            anno_labels = list(sorted(labels[label_key], key=lambda x: x[0]))
            if not _pass_conflict_check(anno_labels):
                raise Exception("标注冲突，数据为：" + json.dumps(item, ensure_ascii=False))
            for single_label in anno_labels:
                positions.add(single_label[0])
                positions.add(single_label[1])
        positions = list(sorted(positions))
        shift_left = dict()
        before_position = 0
        camculate = 0
        for position in positions:
            tmp = text[before_position:position]
            clean_tmp = "".join(tokenizer.normalize(tmp).split())
            interv = len(tmp) - len(clean_tmp) + camculate
            shift_left[position] = interv
            before_position = position
            camculate = interv
        clean_example = {"labels": dict()}
        for label_key in labels:
            anno_labels = list(sorted(labels[label_key], key=lambda x: x[0]))
            for single_label in anno_labels:
                clean_example['labels'][label_key] = clean_example['labels'].get(label_key, [])
                clean_example['labels'][label_key].append(
                    [single_label[0] - shift_left[single_label[0]], single_label[1] - shift_left[single_label[1]],
                     single_label[2]])
        clean_example['text'] = "".join(tokenizer.normalize(text).split())
        clean_data.append(clean_example)
    convert_to_conll_with_chinese_bert(clean_data, filename, data_dir, suffix)


if __name__ == '__main__':
    data = [
        {"text": "这是一个测试数据集合，测试的是BIO数据集转换是否正常",
         "labels": {"label1": [[0, 2, 'ORG'], [11, 17, 'PER']], "label2": [[0, 2, 'Struct1'], [11, 17, 'Struct2']]}},
        {"text": "这是一个测试数据集合，测试的是BIO数据集转换是否正常",
         "labels": {"label1": [[0, 2, 'ORG'], [11, 17, 'PER']], "label2": [[0, 2, 'Struct1'], [11, 17, 'Struct2']]}}
    ]
    convert_to_conll(data, "conlltest")
