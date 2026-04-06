class Counter:
    def __init__(self, max_count: int):
        self.count = 0
        self.max_count = max_count

    def __call__(self, num=1) -> int:
        current = self.count
        self.count = (self.count + 1) % self.max_count
        return current

    def peek(self) -> int:
        return self.count