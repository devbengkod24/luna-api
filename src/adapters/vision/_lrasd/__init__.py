"""Vendored LR-ASD model definition (Light-weight Active Speaker Detection).

Source: https://github.com/Junhua-Liao/LR-ASD (MIT License, (c) 2025 Liao Junhua).
Only the inference-time nn.Module definitions are vendored here; training,
data-loading, and the bundled S3FD face detector are intentionally omitted —
this codebase supplies face crops from its own InsightFace + ByteTrack pipeline.
"""
