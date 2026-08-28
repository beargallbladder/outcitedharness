import os


def pytest_configure() -> None:
    # Unit tests must never discover or mutate the live GCI service via .env.
    os.environ["HARNESS_GCI_TOKEN"] = ""
