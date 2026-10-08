# Lecture 2: Large Language Models

**Status: roadmap and package structure only.** The repository does not yet generate ticket responses. Lecture 1's classifier and API remain the working application.

## Target workflow

```text
Customer support ticket
  → Transformer classification (Lecture 1)
  → LLM draft response (Lecture 2)
  → Human review and approval
  → Support ticket system
```

The draft must not be sent or written to a ticket system before a human approves it. Classification labels and the original ticket can provide context for drafting, but the generator must not invent account status, policy, or resolution steps. The integration phase needs a reviewable draft state and a record of the human decision.

## Implementation order

| Phase | Topic | Implementation target | Check before advancing |
| --- | --- | --- | --- |
| 2.1 | LLM architectures | Compare bidirectional encoder classification with causal decoder generation; build a tiny decoder that predicts the next token | Verify causal masking and output shapes |
| 2.2 | Mixture of Experts | Add a small routed feed-forward block beside a dense baseline | Check routing counts, capacity behavior, and deterministic tests |
| 2.3 | MHA / MQA / GQA | Implement multi-head, multi-query, and grouped-query attention behind one interface | Compare tensor shapes, masking, parameter counts, and outputs |
| 2.4 | RoPE | Rotate query and key vectors by token position | Test position dependence and compatibility with cached decoding |
| 2.5 | Context length | Add incremental decoding and a KV cache; measure memory and latency across lengths | Compare cached and uncached outputs with the same weights |
| 2.6 | Sampling | Add temperature, top-k, and top-p sampling with a fixed seed | Check filtering, reproducibility, and invalid parameter handling |
| 2.7 | Integration | Produce an unsent ticket-response draft using the classified label and ticket text | Evaluate draft quality and enforce human approval before any external write |

The package locations are in [`src/llm/`](../../src/llm/README.md). New config files belong in [`configs/llm/`](../../configs/llm/README.md), and phase tests in [`tests/llm/`](../../tests/llm/README.md). Reuse synthetic tickets while learning; representative, de-identified real tickets and a separate response-quality evaluation are required before production use.

## First implementation milestone

Start 2.1 with a minimal causal self-attention decoder and a small next-token task on synthetic text. Document how its causal mask differs from the bidirectional encoder in [`src/transformer/`](../../src/transformer/). Keep generation experiments separate from the existing `/predict` classification endpoint.
