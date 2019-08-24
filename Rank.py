#!/usr/bin/env python3

from collections import defaultdict
from itertools import combinations
import numpy as np

class Rank:
    """ This class defines an interface for all Ranking systems """
    def __init__(self):
        raise NotImplemented

    def update(self, *args):
        raise NotImplemented

    def ranks(self, *args, **kwargs):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def __str__(self):
        return str(self.rank)

    def __repr__(self):
        return repr(self.rank)

class PointRank(Rank):
    """
    This class defines a ranking system based on points
    """
    def __init__(self):
        self._dirty = False
        self._ranks = []
        super().__init__()

    def update(self, player1, player1_points, player2, player2_points):
        if None not in [player1, player2]:
            super().update(player1, player1_points, player2, player2_points)
        self._dirty = True

    def ranks(self, *args, **kwargs):
        if self._dirty:
            self._ranks = super().ranks(*args, **kwargs)
        self._dirty = False
        return self._ranks

class WinRank(PointRank):
    """
    This class defines a ranking system based on wins and losses
    """
    def update(self, winner, loser):
        super().update(winner, 1, loser, 0)

class RatingRank(PointRank):
    """
    This class defines a ranking system off comparison ratings
    This is essentially a score ranking but scores are normalized based on reporter
    """
    def __init__(self):
        self.record = defaultdict(list)
        super().__init__()

    def update(self, name, score, reporter):
        super().update(None, 0, None, 0)
        self.record[reporter].append((name,score))
        # Super called with to indicate not to use data

    def ranks(self, *args, **kwargs):
        def normalize(x):
            return 8 * (x - 0.5) ** 3 + 0.5
        if self._dirty:
            super().reset()
            for reporter, results in self.record.items():
                max_score = max(a[1] for a in results)
                min_score = max(a[1] for a in results)
                if max_score == min_score:
                    continue
                for (name1, result1), (name2, result2) in combinations(results, 2):
                    score1 = normalize((result1 - min_score) / (max_score - min_score) * .8 + .1)
                    score2 = normalize((result2 - min_score) / (max_score - min_score) * .8 + .1)
                    super().update(name1, score1, name2, score2)
        return super().ranks(*args, **kwargs)

class RankAlgorithm(Rank):
    def __init__(self):
        pass

class PointsPerGameRank(RankAlgorithm):
    """
    Class Calculates rankings based on points per game
    """
    def __init__(self):
        self._data = {}
        super().__init__()

    def update(self, player1, player1_points, player2, player2_points):
        total_points, total_games = self._data.get(player1, (0, 0))
        self._data[player1] = (total_points + player1_points, total_games + 1)
        total_points, total_games = self._data.get(player2, (0, 0))
        self._data[player2] = (total_points + player2_points, total_games + 1)

    def ranks(self):
        raw_ranks = [(name, points/ games) for name, (points, games) in self._data.items()]
        self._ranks = sorted(raw_ranks, key = lambda a: a[1], reverse = True)
        return self._ranks


class PageRank(RankAlgorithm):
    def __init__(self):
        self._data = np.array([], dtype=int)
        self._ids = defaultdict(list)

    def update(self, winner, winner_points, loser, loser_points):
        old_len = len(self._ids)
        winner_id = self._ids[winner] = self._ids.get(winner, len(self._ids))
        loser_id  = self._ids[loser]  = self._ids.get(loser,  len(self._ids))
        N = len(self._ids)

        if old_len < N:
            fill_func = lambda i, j: self._data[i][j] if i < old_len and j < old_len else 0
            self._data = np.fromfunction(np.vectorize(fill_func), (N, N), dtype=int)

        self._data[winner_id][loser_id] += winner_points
        self._data[loser_id][winner_id] += loser_points

    def ranks(self, eps=1.0e-8, d=0.85):
        N = len(self._data)
        fill_func = lambda i,j: self._data[i][j] / (self._data[i][j] + self._data[j][i] + 0.01)
        M = np.fromfunction(np.vectorize(fill_func), (N, N), dtype=int)

        np.fill_diagonal(M, 0.01)
        M = M / M.sum(axis=0)

        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
        last_v = np.ones((N, 1), dtype=np.float32) * 100

        while np.linalg.norm(v - last_v, 2) > eps:
            last_v = v
            v = d * np.matmul(M, v) + (1 - d) / N
        v = v.transpose()[0]
        scores = sorted(zip(self._ids, v), key = lambda a: a[1], reverse = True)
        return scores

class LadderRank(RankAlgorithm):
    def __init__(self):
        self._ranks = []

    def update(self, winner, loser):

        find_in_list = lambda a, l: l.index(a) if a in l else (len(l), l.append(a))[0]
        winner_rank = find_in_list(winner, self.ranks)
        loser_rank = find_in_list(loser, self.ranks)

        if loser_rank > winner_rank:
            new_rank = (loser_rank + winner_rank) // 2
            self.ranks.remove(winner_rank)
            self.ranks.insert(winner, new_rank)

    def ranks(self):
        return self.ranks



class ELORank(RankAlgorithm):
    pass

class PPGRatingTRanks(RatingRank, PointsPerGameRank):
    pass

class WinPageRank(WinRank, PageRank):
    pass

