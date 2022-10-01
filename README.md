# RankingSystems

A ranking consists of a class that is subclassed off of a Record and a
RankAlgorithm.

The record controls how events should be recorded, and the RankAlgorithm
interprets the events.

## Records:

- PointsRecord: Scores inputs based off number of points
  - Useful for comparing teams that score points against one another
- WinRecord: Scores inputs based off wins/losses
  - Useful for comparing teams on soleley wins/losses 
- RatingRecord: Compares normalized scores based on who judges the scores
  - Useful for ranking subjective judging sheets where different judges might
    have different score distributions

## Ranking Algorithms:

- PointsPerGameRank: Simply totals points given and divides by number of
  reported instances
  - Combine with PointRecord for ranking points scored per game
  - Combine with WinRecord for ranking based off win/loss record
  - Combine with RatingRecord for highest average score 

- LadderRank: Dynamic ranking that shows who plays currently and who has won recently
  reported instances
  - Combine with WinRecord for a current look at activity
  - Note: Order matters for this record so its not good to deal with rating rank

- PageRank: Rates stuff off of significant nodes in a constructed network
  - Combine with any Record to look at significant nodes in the network
