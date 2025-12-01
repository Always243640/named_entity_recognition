from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="./DataNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split + ".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            # 先去除两端空白字符
            line = line.strip()

            # 如果是空行，表示一个句子的结束
            if not line:
                if word_list:  # 确保word_list不为空
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []
                continue

            # 分割单词和标签
            parts = line.split()
            if len(parts) == 2:
                word, tag = parts
                word_list.append(word)
                tag_list.append(tag)
            else:
                print(f"警告：跳过格式错误的行: {line}")
                continue

        # 处理文件末尾没有空行的情况
        if word_list:
            word_lists.append(word_list)
            tag_lists.append(tag_list)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
