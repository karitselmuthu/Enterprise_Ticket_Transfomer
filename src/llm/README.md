# Lecture 2: Large Language Models

This package reserves the implementation areas for Lecture 2. No response generator is implemented yet. The existing ticket classifier remains in `src/transformer/` and the other Lecture 1 packages.

| Package | Phase | Planned work |
| --- | --- | --- |
| `architectures/` | 2.1 | Encoder and decoder comparison; a small causal decoder |
| `moe/` | 2.2 | Lightweight expert routing and load diagnostics |
| `attention/` | 2.3 | MHA, MQA, and GQA with shared shape tests |
| `positional_encoding/` | 2.4 | Rotary position embeddings (RoPE) |
| `context/` | 2.5 | Context-window experiments and KV caching |
| `sampling/` | 2.6 | Temperature, top-k, and top-p token selection |
| `inference/` | 2.7 | Ticket-response drafting with human review |

Use the [Lecture 2 roadmap](../../docs/lecture_02_llm/README.md) to implement these in order. Keep educational model code separate from `api/` until the draft and approval workflow has its own evaluation and access controls.
