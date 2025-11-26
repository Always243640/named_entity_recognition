from collections import Counter

from utils import flatten_lists


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

        # 按实体类型聚合的评价指标
        self.entity_types = self._extract_entity_types()
        self.entity_correct_counts, self.entity_golden_counts, \
            self.entity_predict_counts = self._aggregate_entity_counts()
        self.entity_precision = {}
        self.entity_recall = {}
        self.entity_f1 = {}
        self.overall_precision = 0.
        self.overall_recall = 0.
        self.overall_f1 = 0.
        self._cal_entity_level_metrics()

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['标签', 'precision', 'recall', 'f1-score', 'support']
        print(header_format.format(*header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印总体指标
        print(row_format.format(
            'overall',
            self.overall_precision,
            self.overall_recall,
            self.overall_f1,
            sum(self.entity_golden_counts.values())
        ))

        # 打印每类实体的指标
        for entity in sorted(self.entity_types):
            print(row_format.format(
                entity,
                self.entity_precision[entity],
                self.entity_recall[entity],
                self.entity_f1[entity],
                self.entity_golden_counts[entity]
            ))

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _extract_entity_type(self, tag):
        if tag == 'O':
            return None
        try:
            return tag.split('-')[1]
        except IndexError:
            return None

    def _extract_entity_types(self):
        entity_types = set()
        for tag in self.tagset:
            entity = self._extract_entity_type(tag)
            if entity:
                entity_types.add(entity)
        return entity_types

    def _aggregate_entity_counts(self):
        correct = Counter()
        golden = Counter()
        predict = Counter()

        for tag, count in self.correct_tags_number.items():
            entity = self._extract_entity_type(tag)
            if entity:
                correct[entity] += count

        for tag, count in self.golden_tags_counter.items():
            entity = self._extract_entity_type(tag)
            if entity:
                golden[entity] += count

        for tag, count in self.predict_tags_counter.items():
            entity = self._extract_entity_type(tag)
            if entity:
                predict[entity] += count

        return correct, golden, predict

    def _cal_entity_level_metrics(self):
        eps = 1e-10
        total_correct = sum(self.entity_correct_counts.values())
        total_predict = sum(self.entity_predict_counts.values())
        total_golden = sum(self.entity_golden_counts.values())

        self.overall_precision = total_correct / (total_predict + eps)
        self.overall_recall = total_correct / (total_golden + eps)
        self.overall_f1 = 2 * self.overall_precision * self.overall_recall / \
            (self.overall_precision + self.overall_recall + eps)

        for entity in self.entity_types:
            correct = self.entity_correct_counts[entity]
            predict = self.entity_predict_counts[entity]
            golden = self.entity_golden_counts[entity]

            precision = correct / (predict + eps)
            recall = correct / (golden + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            self.entity_precision[entity] = precision
            self.entity_recall[entity] = recall
            self.entity_f1[entity] = f1

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))
