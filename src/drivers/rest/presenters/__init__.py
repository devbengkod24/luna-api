"""Presenters convert domain events / entities into wire-format dicts."""

from .stream import stream_event_to_dict

__all__ = ["stream_event_to_dict"]
