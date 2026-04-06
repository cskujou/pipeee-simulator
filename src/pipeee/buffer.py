class Buffer:
    def __init__(self, num_buffers: int):
        self.num_buffers = num_buffers
        self.buffers = [[] for _ in range(num_buffers)]

    def add_to_buffer(self, buffer_idx: int, item):
        self.buffers[buffer_idx].append(item)

    def get_from_buffer(self, buffer_idx: int):
        if self.buffers[buffer_idx]:
            return self.buffers[buffer_idx].pop(0)
        else:
            raise IndexError("Buffer is empty")

    def is_buffer_empty(self, buffer_idx: int) -> bool:
        return len(self.buffers[buffer_idx]) == 0