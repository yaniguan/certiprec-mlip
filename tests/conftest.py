"""Tiny contraction tests run faster and more consistently with one CPU thread."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
