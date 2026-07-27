from concurrent.futures import FIRST_COMPLETED, wait


def iter_bounded_futures(executor, function, tasks, max_pending):
    """Yield completed futures while retaining only a bounded task queue."""
    if max_pending < 1:
        raise ValueError("max_pending must be at least 1")

    task_iterator = iter(tasks)
    in_flight = {}

    def submit_next():
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        in_flight[executor.submit(function, task)] = task
        return True

    while len(in_flight) < max_pending and submit_next():
        pass

    while in_flight:
        completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in completed:
            task = in_flight.pop(future)
            yield future, task
            submit_next()
