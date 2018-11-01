

class Sequence(object):
    """Represents a complete or partial sequence."""
    def __init__(self, sentence, state, logprob, last_token, score=0, number_epsilons=0):
        """Initializes the Sequence.
        Args:
          sentence: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.last_token = last_token
        self.number_epsilons = number_epsilons

    def __cmp__(self, other):
        """Compares Sequences by logprob."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class SeqSet(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def append(self, x):
        assert self._data is not None
        self._data.append(x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = []
        if sort:
            data.sort(reverse=True)
        return data


