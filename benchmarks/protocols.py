def get_eval_stream(benchmark, current_experience_id: int, protocol: str):
    """
    protocol='fixed':
        Evaluate on all test experiences after every training experience.

    protocol='growing':
        Evaluate only on test experiences seen so far.
    """

    protocol = protocol.lower()

    if protocol == "fixed":
        return benchmark.test_stream

    if protocol == "growing":
        return benchmark.test_stream[: current_experience_id + 1]

    raise ValueError("protocol must be either 'fixed' or 'growing'")