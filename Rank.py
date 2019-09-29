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
    """ This class defines a ranking system based on points """
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
    """ This class defines a ranking system based on wins and losses """
    def update(self, winner, loser):
        super().update(winner, 1, loser, 0)


class RatingRank(PointRank):
    """
    This class defines a ranking system off comparison ratings
    This is essentially a score ranking but
    scores are normalized based on reporter
    """
    def __init__(self):
        self.record = defaultdict(list)
        super().__init__()

    def update(self, name, score, reporter):
        super().update(None, 0, None, 0)
        self.record[reporter].append((name, score))
        # Super called with to indicate not to use data

    def ranks(self, *args, **kwargs):
        def normalize(x):
            return 8 * (x - 0.5) ** 3 + 0.5
        if self._dirty:
            super().reset()
            for reporter, results in self.record.items():
                max_s = max(a[1] for a in results)
                min_s = min(a[1] for a in results)
                if max_s == min_s:
                    continue
                for (n1, r1), (n2, r2) in combinations(results, 2):
                    s1 = normalize((r1 - min_s) / (max_s - min_s) * .8 + .1)
                    s2 = normalize((r2 - min_s) / (max_s - min_s) * .8 + .1)
                    super().update(n1, s1, n2, s2)
        return super().ranks(*args, **kwargs)


class RankAlgorithm(Rank):
    def __init__(self):
        pass


class PointsPerGameRank(RankAlgorithm):
    """ Class Calculates rankings based on points per game """
    def __init__(self):
        self._data = {}
        super().__init__()

    def update(self, player1, player1_points, player2, player2_points):
        total_points, total_games = self._data.get(player1, (0, 0))
        self._data[player1] = (total_points + player1_points, total_games + 1)
        total_points, total_games = self._data.get(player2, (0, 0))
        self._data[player2] = (total_points + player2_points, total_games + 1)

    def ranks(self):
        raw_ranks = [(name, points / games)
                     for name, (points, games) in self._data.items()]
        self._ranks = sorted(raw_ranks, key=lambda a: a[1], reverse=True)
        return self._ranks


class PageRank(RankAlgorithm):
    """
    Page rank builds a graph based on wins and losses and run the page rank
    algorithm to find the most significant nodes. This fairly elegantly figures
    out loops based on extraneous information. Works best for connected groups
    although does an ok job for semi isolated populations if there are at least
    a few connections. The d parameter is roughly 1 - the minimum chance that
    in any random game one player will beat another. In the literature this
    number is usually 0.85 but for competitive tournaments this should probably
    be about 0.95 - 0.99.
    """
    def __init__(self):
        self._data = np.array([], dtype=int)
        self._ids = defaultdict(list)

    def update(self, winner, winner_points, loser, loser_points):
        old_len = len(self._ids)
        winner_id = self._ids[winner] = self._ids.get(winner, len(self._ids))
        loser_id = self._ids[loser] = self._ids.get(loser,  len(self._ids))
        N = len(self._ids)

        if old_len < N:
            new_data = np.zeros((N, N))
            new_data[:old_len, :old_len] += self._data
            self._data = new_data

        self._data[winner_id][loser_id] += winner_points
        self._data[loser_id][winner_id] += loser_points

    def ranks(self, eps=1.0e-8, d=0.85):
        N = len(self._data)

        def fill_func(i, j, e=0.01):
            return self._data[i][j] / (self._data[i][j] + self._data[j][i] + e)
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
        scores = sorted(zip(self._ids, v), key=lambda a: a[1], reverse=True)
        return scores


class LadderRank(RankAlgorithm):
    """
    Ladder Rank is a dynamic ranking system where when any player beats a
    player of higher rank they move up half the distance to that player.
    Its benefits are that players who play a lot are more likely to move up and
    no player is discouraged from palying a lower player
    """
    def __init__(self):
        self._ranks = []

    def update(self, winner, loser):
        def find_in_list(a, l):
            if a in l:
                return l.index(a)
            l.append(a)
            return len(l)-1
        winner_rank = find_in_list(winner, self.ranks)
        loser_rank = find_in_list(loser, self.ranks)

        if loser_rank > winner_rank:
            new_rank = (loser_rank + winner_rank) // 2
            self.ranks.remove(winner_rank)
            self.ranks.insert(winner, new_rank)

    def ranks(self):
        return self.ranks


class PPGRatingTRanks(RatingRank, PointsPerGameRank):
    pass


class WinPageRank(WinRank, PageRank):
    pass
