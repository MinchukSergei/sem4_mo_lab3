import itertools
from random import shuffle
from tqdm import tqdm
import time


class GridSearch:
    def __init__(self, nn, hp, fit, metric, rnd_count=None):
        self.nn = nn
        self.hp = hp
        self.models = []
        self.fit = fit
        self.rnd_count = rnd_count
        self.results = []
        self.metric = metric

        self.cartesian_product_hp()

    def cartesian_product_hp(self):
        hp_names = self.hp.keys()

        print('Model computing:')
        for hp in tqdm(itertools.product(*self.hp.values())):
            model = {}

            for n, hp_v in zip(hp_names, hp):
                model[n] = hp_v

            self.models.append(model)

    def execute(self):
        models = self.models.copy()

        if self.rnd_count is not None:
            shuffle(models)
            models = models[:self.rnd_count]

        for m in tqdm(models):
            start_time = time.perf_counter()
            result = self.fit(self.nn, m)
            elapsed_time = time.perf_counter() - start_time

            self.results.append({
                'model': m,
                'result': result,
                'time': elapsed_time
            })

    def get_total_time(self):
        return sum(list(map(lambda t: t['time'], self.results)))

    def get_results(self, desc=True):
        return sorted(self.results, reverse=desc, key=lambda a: a['result'][self.metric])
