class GroverReturn:
    counts = None
    indices = None

    def __init__(self, counts, indices):
        self.counts = counts
        self.indices = indices

    def to_dict(self):
        return {'Counts': self.counts, 'Indices': self.indices}

    def __str__(self):
        return str(self.to_dict())
