

class IdPool:
    """
    通用的 ID 管理器，负责高效地分配和回收唯一 ID

    支持两种模式：
    1. 受限模式：有最大容量限制，当达到限制时会抛出异常
    2. 无限模式：无容量限制，ID 会一直递增

    该管理器通过维护可用 ID 池来实现高效的 ID 复用，防止 ID 冲突
    """
    def __init__(self, max_size: int = None, initial_capacity: int = 0):
        """
        初始化 ID 管理器

        Args:
            max_size: 最大容量限制。如果是 None，则表示无限模式
            initial_capacity: 初始可用 ID 池的大小（仅在 max_size 为 None 时有效）
        """
        self.max_size = max_size
        self.available_ids = set()
        self.used_ids = set()
        self.next_new_id = 0

        if max_size is not None:
            # 受限模式：预分配所有可用 ID
            self.available_ids = set(range(max_size))
        elif initial_capacity > 0:
            # 无限模式：预分配初始容量的可用 ID
            self.available_ids = set(range(initial_capacity))
            self.next_new_id = initial_capacity

    def acquire(self) -> int:
        """
        获取一个可用的 ID

        Returns:
            可用的 ID

        Raises:
            ValueError: 当受限模式下没有可用 ID 时抛出异常
        """
        if self.available_ids:
            # 从可用池获取 ID
            id_ = self.available_ids.pop()
        else:
            # 无可用 ID，需要分配新 ID
            if self.max_size is not None:
                raise ValueError("No available IDs in restricted mode pool")
            id_ = self.next_new_id
            self.next_new_id += 1

        self.used_ids.add(id_)
        return id_

    def release(self, id_: int):
        """
        释放一个 ID，将其添加到可用池

        Args:
            id_: 要释放的 ID
        """
        if id_ not in self.used_ids:
            return

        self.used_ids.remove(id_)

        if self.max_size is None or (0 <= id_ < self.max_size):
            self.available_ids.add(id_)

    def reset(self):
        """
        重置 ID 管理器，释放所有已使用的 ID
        """
        if self.max_size is not None:
            # 受限模式：重置为初始状态
            self.available_ids = set(range(self.max_size))
        else:
            # 无限模式：保留已分配过的所有 ID 到可用池
            self.available_ids = set(range(self.next_new_id))
        self.used_ids.clear()

    def get_used_count(self) -> int:
        """
        获取当前使用中的 ID 数量

        Returns:
            使用中的 ID 数量
        """
        return len(self.used_ids)

    def get_available_count(self) -> int:
        """
        获取可用的 ID 数量

        Returns:
            可用的 ID 数量
        """
        return len(self.available_ids)

    def is_id_used(self, id_: int) -> bool:
        """
        检查一个 ID 是否已被使用

        Args:
            id_: 要检查的 ID

        Returns:
            是否已被使用
        """
        return id_ in self.used_ids

    def get_all_used_ids(self) -> list[int]:
        """
        获取所有已使用的 ID 列表

        Returns:
            已使用的 ID 列表
        """
        return list(self.used_ids)
