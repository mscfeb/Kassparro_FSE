from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class TransientProviderError(Exception):
    pass


class PermanentProviderError(Exception):
    pass


class ProviderTimeoutError(TimeoutError):
    pass


_GLOBAL_PROVIDER_SEMAPHORE = asyncio.Semaphore(3)


class RequestExecutor:
    def __init__(
        self,
        timeout_seconds: float = 15.0,
        retry_delay_seconds: float = 0.1,
        max_attempts: int = 2,
        semaphore: asyncio.Semaphore | None = None,
    ):
        if max_attempts not in (1, 2):
            raise ValueError("max_attempts must be one or two")
        self.timeout_seconds = timeout_seconds
        self.retry_delay_seconds = min(max(retry_delay_seconds, 0.0), 1.0)
        self.max_attempts = max_attempts
        self.semaphore = semaphore

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(self.max_attempts):
            try:
                async with _GLOBAL_PROVIDER_SEMAPHORE:
                    if self.semaphore is None:
                        return await asyncio.wait_for(operation(), timeout=self.timeout_seconds)
                    async with self.semaphore:
                        return await asyncio.wait_for(operation(), timeout=self.timeout_seconds)
            except (asyncio.TimeoutError, TransientProviderError) as exc:
                if attempt + 1 >= self.max_attempts:
                    if isinstance(exc, asyncio.TimeoutError):
                        raise ProviderTimeoutError("provider request timed out") from exc
                    raise
                await asyncio.sleep(self.retry_delay_seconds)
        raise AssertionError("unreachable retry state")


async def gather_isolated(
    items: list[T],
    worker: Callable[[T], Awaitable[object]],
) -> list[tuple[T, object | BaseException]]:
    async def run(item: T) -> tuple[T, object | BaseException]:
        try:
            return item, await worker(item)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            return item, exc

    results = await asyncio.gather(*(run(item) for item in items))
    return results