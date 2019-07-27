#!/usr/bin/env python3

from collections import defaultdict

class Rank:
    """ This class defines an interface for all Ranking systems """
    def __init__(self):
        print("called")
        self.rank = []

    def update(self, *args):
        raise NotImplemented

    def ranks(self):
        raise NotImplemented

    def __str__(self):
        return str(self.rank)

    def __repr__(self):
        return repr(self.rank)


class PointRank(Rank):
    """
    This class defines a ranking system based on points
    """
    pass

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
        ratings = self.record[reporter]
        if len(ratings) = 0:
            pass




class PageRank(Rank):
    def update(self, *args):
        pass

class LadderRank(Rank):
    pass

class ELORank(Rank):
    pass

RatingRank()
