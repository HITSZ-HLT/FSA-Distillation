from utils import append_new_line, save_json
import os, time, json
from datetime import datetime



class F1Measure:
    def __init__(self):
        self.pred_list = []  # List to store predictions
        self.true_list = []  # List to store ground truths

    def add_predictions(self, idx, preds):
        """Adds a batch of predictions for a specific index."""
        self.pred_list.extend((idx, pred) for pred in preds)

    def add_ground_truths(self, idx, trues):
        """Adds a batch of ground truths for a specific index."""
        self.true_list.extend((idx, true) for true in trues)

    def report(self):
        """Calculates and returns the F1 score."""
        self.f1, self.precision, self.recall = self.calculate_f1()
        return self.f1

    def __getitem__(self, key):
        """Allows retrieval of attributes like a dictionary."""
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{key} is not a valid attribute of F1Measure.")

    def calculate_f1(self):
        """Calculates F1 score along with precision and recall."""
        n_tp = sum(pred in self.true_list for pred in self.pred_list)
        precision = n_tp / len(self.pred_list) if self.pred_list else 1

        n_tp = sum(true in self.pred_list for true in self.true_list)
        recall = n_tp / len(self.true_list) if self.true_list else 1

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return f1, precision, recall



def parse_aspect_polarity(atsa_seq, sentence):
    aspect_polarity_list = []
    valid_flag = True

    def parse_seq(seq):
        if seq.count('|') != 1:
            return False
        
        aspect, polarity = seq.split('|')
        aspect = aspect.strip()
        polarity = polarity.strip()

        if aspect not in sentence:
            return False 

        if polarity not in ('positive', 'neutral', 'negative', 'conflict'):
            return False

        return aspect, polarity

    for sub_seq in atsa_seq.split(';'):
        sub_seq = sub_seq.strip()
        aspect_polarity = parse_seq(sub_seq)
        if not aspect_polarity:
            valid_flag = False
        else:
            aspect_polarity_list.append(aspect_polarity)

    return aspect_polarity_list, valid_flag



class Result:
    def __init__(self, data):
        self.data = data 

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, outputs):
        data = {}

        ID = 0
        for output in outputs:
            examples = output['examples']
            predictions = output['predictions']

            for example, prediction in zip(examples, predictions):
                sentence = example['sentence']

                aspect_polarity_list_true = parse_aspect_polarity(example['atsa_seq'], sentence)[0]
                aspect_polarity_list_pred = parse_aspect_polarity(prediction, sentence)[0]

                data[ID] = {
                    'ID': example.get('ID', ID),
                    'sentence': sentence,
                    'aspect_polarity': aspect_polarity_list_true,
                    'prediction': aspect_polarity_list_pred,
                    'prediction_string': example.get('prediction_string', None)
                }
                ID += 1

        return cls(data)

    def cal_metric(self):
        f1 = F1Measure()

        for ID in self.data:
            example = self.data[ID]
            f1.add_ground_truths(ID, example['aspect_polarity'])
            f1.add_predictions(ID, example['prediction'])

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['recall'],
            'precision': f1['precision'],
        }

        self.monitor = self.detailed_metrics['f1']

    def save_prediction(self, output_dir, model_name_or_path, subname, dataset, seed, lr):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        file_name = os.path.join(output_dir, now, f'{dataset}_{subname}_{seed}.json')

        print('save prediction to', file_name)
        save_json(
            {
                'data': self.data,
                'meta': (model_name_or_path, subname, dataset, seed, lr, now)
            }, 
            file_name
        )

    def save_metric(self, output_dir, model_name_or_path, subname, dataset, seed, lr):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        performance_file_name = os.path.join(output_dir, now, 'performance.txt')

        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'subname': subname,
            'dataset': dataset,
            'seed': seed,
            'lr': lr,
            'metric': self.detailed_metrics
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
