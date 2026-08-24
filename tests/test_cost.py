from harness.config import Pricing
from harness.cost import estimate_cost


def test_zero_local_cost():
    assert estimate_cost(Pricing(input_per_million=0, output_per_million=0), 1000, 50) == 0


def test_unknown_price_is_none():
    assert estimate_cost(Pricing(), 1000, 50) is None


def test_known_price():
    cost = estimate_cost(
        Pricing(input_per_million=1.0, output_per_million=5.0),
        1_000_000,
        1_000_000,
    )
    assert cost == 6.0
