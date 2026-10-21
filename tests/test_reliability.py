import asyncio

import pytest

from app.reliability import ProviderTimeoutError, RequestExecutor, TransientProviderError, gather_isolated


@pytest.mark.asyncio
async def test_timeout_is_retried_once_then_reported():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.05)
        return "ok"

    result = await RequestExecutor(timeout_seconds=0.02, retry_delay_seconds=0, semaphore=asyncio.Semaphore(1)).call(operation)
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_transient_failure_retries_but_permanent_failure_does_not():
    calls = 0

    async def transient():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientProviderError("try again")
        return 3

    assert await RequestExecutor(retry_delay_seconds=0).call(transient) == 3
    assert calls == 2

    async def permanent():
        nonlocal calls
        calls += 1
        raise ValueError("permanent")

    before = calls
    with pytest.raises(ValueError):
        await RequestExecutor(retry_delay_seconds=0).call(permanent)
    assert calls == before + 1


@pytest.mark.asyncio
async def test_timeout_exhaustion_returns_sanitized_error():
    async def operation():
        await asyncio.sleep(0.05)

    with pytest.raises(ProviderTimeoutError):
        await RequestExecutor(timeout_seconds=0.001, retry_delay_seconds=0).call(operation)


@pytest.mark.asyncio
async def test_global_semaphore_bounds_concurrent_requests():
    active = 0
    peak = 0
    lock = asyncio.Lock()
    executor = RequestExecutor(timeout_seconds=1, retry_delay_seconds=0, semaphore=asyncio.Semaphore(3))

    async def worker(_):
        nonlocal active, peak

        async def operation():
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1

        await executor.call(operation)

    await asyncio.gather(*(worker(index) for index in range(12)))
    assert peak == 3


@pytest.mark.asyncio
async def test_failure_isolation_preserves_other_results():
    async def worker(value):
        if value == 2:
            raise RuntimeError("one answer failed")
        return value * 2

    results = await gather_isolated([1, 2, 3], worker)
    assert results[0] == (1, 2)
    assert results[1][0] == 2 and isinstance(results[1][1], RuntimeError)
    assert results[2] == (3, 6)