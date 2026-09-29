"""TruthLens: training-free deepfake image verification via VQA-style probing.

Pipeline (paper Section 2): probe prompts P -> LVLM answers A -> summary S = g(A)
-> LLM verdict and justification (y, r) = f_LM(S).
"""
__version__ = "1.1.0"
