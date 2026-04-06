class SchedulerBudgetFactory:
    def __init__(self, sche_budget_config):
        budget_cls = sche_budget_config.scheduler_budget_cls
        self.budget_cls = BUDGET_CLASS_MAP[budget_cls] if isinstance(budget_cls, str) else budget_cls
        self.config = sche_budget_config

    def __call__(self) -> "SchedulerBudgetBase":
        return self.budget_cls(self.config)


class SchedulerBudgetBase:
    def __init__(self, config):
        raise NotImplementedError()

    def pre_check(self) -> bool:
        raise NotImplementedError()

    def post_check(self, request) -> bool:
        raise NotImplementedError()

    def consume(self):
        raise NotImplementedError()

    def release(self, request):
        raise NotImplementedError()

class DefaultSchedulerBudget:
    def __init__(self, config):
        self.batch_size_budget = config.max_batch_size

    def pre_check(self) -> bool:
        return self.batch_size_budget > 0

    def post_check(self, request) -> bool:
        return True

    def consume(self):
        self.batch_size_budget -= 1
        
    def release(self, request):
        self.batch_size_budget += 1


class BMCSchedulerBudget:
    def __init__(self, config):
        self.batch_size_budget = config.max_batch_size
        self.memory_budget = config.max_memory_budget
        self.compute_budget = config.max_compute_budget

    def pre_check(self):
        # TODO: implement BMC pre-check logic
        return self.batch_size_budget > 0

    def post_check(self, request):
        # TODO: implement BMC post-check logic
        pass

    def consume(self):
        self.batch_size_budget -= 1

    def release(self, request):
        self.batch_size_budget += 1


BUDGET_CLASS_MAP = {
    "default": DefaultSchedulerBudget,
    "bmc": BMCSchedulerBudget,
}
