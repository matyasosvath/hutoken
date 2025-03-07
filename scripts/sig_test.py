from random import shuffle
from statistics import mean


class DiffMeansPermute(object):

    def __init__(self, data) -> None:
        self.data = data
        self.make_model()
        self.actual = self.test_statistic(data)

    def p_value(self, iters: int = 1000) -> float:
        self.test_stats = [self.test_statistic(self.run_model()) for _ in range(iters)]
        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def test_statistic(self, data: tuple[list, list]) -> float:
        group1, group2 = data
        test_stat = abs(mean(group1) - mean(group2)) # two-sided
        return test_stat

    def make_model(self) -> None:
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = group1 + group2

    def run_model(self) -> tuple[list, list]:
        shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data