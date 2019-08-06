#!/usr/bin/env python3

from collections import defaultdict
from itertools import combinations
import numpy

class Rank:
    """ This class defines an interface for all Ranking systems """
    def __init__(self):
        raise NotImplemented

    def update(self, *args):
        raise NotImplemented

    def ranks(self):
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
        self.ranks = []

    def update(self, player1, player1_points, player2, player2_points):
        if None not in [player1, player2]:
            super().update(player1, player1_points, player2, player2_points)
        self._dirty = True

    def ranks(self):
        if self._dirty:
            self.ranks = super().ranks()
        self._dirty = False
        return self.ranks

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

    def update(self, name, score, reporter):
        super().update(None, 0, None, 0)
        self.record[reporter].append((name,score))
        # Super called with to indicate not to use data

    def ranks(self):
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
        return super().ranks()

class RankAlgorithm(Rank):
    pass

class PointsPerGameRank(RankAlgorithm):
    """
    Class Calculates rankings based on points per game
    """
    def __init__(self):
        self.data = {}
        super().__init__()

    def update(self, player1, player1_points, player2, player2_points):
        total_points, total_games = self.data.get(player1, (0, 0))
        self.data[player1] = (total_points + player1_points, total_games + 1)
        total_points, total_games = self.data.get(player2, (0, 0))
        self.data[player2] = (total_points + player2_points, total_games + 1)

    def ranks(self):
        raw_ranks = [(name, points/ games) for name, (points, games) in self.data.items()]
        self.ranks = sorted(raw_ranks, key = lambda a: a[1], reverse = True)
        return self.ranks


class PageRank(RankAlgorithm):
    def update(self, *args):
        pass

class LadderRank(RankAlgorithm):
    pass

class ELORank(RankAlgorithm):
    pass

class PPGRatingTRanks(RatingRank, PointsPerGameRank):
    pass

