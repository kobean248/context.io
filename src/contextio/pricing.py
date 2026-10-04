"""Dollar cost of a request, given token counts and a price sheet."""

from __future__ import annotations

from dataclasses import dataclass

_PER = 1_000_000


@dataclass(frozen=True)
class PricingModel:
    """Prices in dollars per million tokens.

    The defaults are placeholders in the range of current frontier models; pass real numbers for
    the model you are evaluating.
    """

    input_per_mtok: float = 3.0
    output_per_mtok: float = 15.0
    cached_input_per_mtok: float | None = 0.3
    embedding_per_mtok: float = 0.0

    def __post_init__(self) -> None:
        prices = (self.input_per_mtok, self.output_per_mtok, self.embedding_per_mtok)
        if any(p < 0 for p in prices) or (self.cached_input_per_mtok or 0) < 0:
            raise ValueError("prices must be non-negative")

    def cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> float:
        """Total request cost; ``cached_input_tokens`` is the cached subset of ``input_tokens``."""
        if not 0 <= cached_input_tokens <= input_tokens:
            raise ValueError("cached_input_tokens must be between 0 and input_tokens")
        cached_price = (
            self.input_per_mtok
            if self.cached_input_per_mtok is None
            else self.cached_input_per_mtok
        )
        uncached = input_tokens - cached_input_tokens
        return (
            uncached * self.input_per_mtok
            + cached_input_tokens * cached_price
            + output_tokens * self.output_per_mtok
            + embedding_tokens * self.embedding_per_mtok
        ) / _PER
