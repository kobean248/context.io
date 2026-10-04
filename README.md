# context.io

[![CI](https://github.com/kobean248/context.io/actions/workflows/ci.yml/badge.svg)](https://github.com/kobean248/context.io/actions/workflows/ci.yml)

**A benchmark and toolkit for choosing which user context to send to an LLM when you have a token budget.**

> Given a fixed context budget, how do we select, compress, cache, and retrieve user context to
> maximize task performance per token and per dollar?

A long-lived assistant builds up thousands of pieces of user context: preferences, projects,
decisions, past conversations. Sending all of it on every request is slow and expensive.
Retrieving too little drops the facts the answer depends on. contextio treats this as an
engineering optimization problem that you can measure. You build strategies, run them on users
with known ground truth, and compare accuracy against tokens, latency, and cost.

The core library has zero runtime dependencies and runs fully offline.

## Problem formulation

Let $C$ be a user's available context, $q$ a query, and $R(q, C)$ the context a strategy
selects. We want

$$\max_R \; \text{Performance}(q, R(q, C)) \quad \text{subject to} \quad \text{Tokens}(R(q, C)) \le B$$

where $B$ is the context budget. Each memory has a weight (its token count) and a value (its
relevance), so selection is a knapsack problem.

## Architecture

```text
                 User request
                      │
                      ▼
   Layer 1: Memory      raw conversations → facts → dedup → store      (consolidation.py)
                      │
                      ▼
   Layer 2: Retrieval   recency · BM25 · vector · hybrid scoring        (retrieval/)
                      │
                      ▼
   Layer 3: Context     budgeted selection · ordering · prefix caching  (selection.py, cache.py)
                      │
                      ▼
                 LLM / Agent  ──►  metrics: recall, precision, utilization, tokens, latency, $
```

| Module | What it provides |
| --- | --- |
| `memory` | `Memory` (static / episodic / semantic), `MemoryStore`, write-time importance heuristic |
| `retrieval` | `RecencyRetriever`, `BM25Retriever`, `VectorRetriever` with a pluggable `Embedder`, `HybridRetriever` |
| `selection` | `GreedySelector`, `DensitySelector`, `KnapsackSelector`, static-first prompt ordering |
| `consolidation` | rule-based and LLM fact extraction, near-duplicate merging, `CompressedStrategy` |
| `cache` | `PrefixCache`, which simulates provider prompt caching of stable prefixes |
| `metrics`, `pricing` | context precision, recall, utilization, stale leakage, efficiency, and $ cost |
| `bench` | deterministic synthetic users with labeled ground truth |
| `eval` | budget-sweep runner, summaries, accuracy-vs-tokens frontier, optional LLM grading |

### Hybrid scoring

`HybridRetriever` scores each memory as

$$S(m, q) = w_r R(m) + w_s \text{Sim}(m, q) + w_i I(m) + w_f F(m)$$

where $R$ is recency decay (static memories never decay), Sim is BM25 plus vector similarity,
$I$ is write-time importance, and $F$ is how often the memory has been used. A similarity
gate stops importance or recency alone from pulling off-topic memories into context.

## Quickstart

```bash
pip install -e ".[dev]"

# Run every strategy on 100 synthetic users across budgets of 250 to 8,000 tokens
contextio run --users 100 --out results/default

# Break results down by query type
contextio run --users 100 --by-kind --budgets 1000

# Save the dataset for reuse or inspection
contextio generate --users 100 --out data/synthetic.jsonl
contextio run --dataset data/synthetic.jsonl

# Grade real model answers (any OpenAI-compatible endpoint, including local Ollama/vLLM)
OPENAI_API_KEY=... contextio run --users 10 --llm-model gpt-4o-mini
```

Or from Python:

```python
from contextio.bench import generate_dataset
from contextio.eval import BenchmarkConfig, run_benchmark, summarize, to_markdown
from contextio.retrieval import HybridRetriever
from contextio.selection import KnapsackSelector
from contextio.strategies import FullContext, RankedStrategy

dataset = generate_dataset(n_users=20)
strategies = [
    FullContext(),
    RankedStrategy("my-hybrid", HybridRetriever(pin_static=True), KnapsackSelector()),
]
records = run_benchmark(dataset, strategies, BenchmarkConfig(budgets=(500, 2000)))
print(to_markdown(summarize(records)))
```

Any object with a `name` and a `select(query, store, budget)` method can be benchmarked.

## The benchmark

`contextio.bench` generates users with about six months of assistant history (~13.9k tokens per
user by default):

- a **static profile**: role, experience, languages, response style, and personal facts
- a **database preference** that a later message supersedes for ~60% of users
- **2-3 projects**, each with an introduction, technical decisions (some revised later), and status updates
- **45-65 distractor sessions** of technical Q&A on overlapping topics, small talk, and personal
  asides that use the same "I prefer" / "we decided" phrasing as real facts

Every query records the memory ids required to answer it, the **stale** ids that would give an
outdated answer, and keywords a correct answer must contain. There are five query types:
`background`, `style`, `preference`, `decision`, and `continuation`.

## Metrics

| Metric | Definition |
| --- | --- |
| answerable rate | fraction of queries where every required memory made it into context |
| recall / precision | required memories covered / selected memories that are relevant |
| utilization | relevant tokens / selected tokens |
| stale rate | fraction of contexts that include superseded information |
| cache hit rate | input tokens served from a simulated prompt-prefix cache |
| answerable per 1k tokens | answerable rate / (mean context tokens / 1000) |
| accuracy | keyword-graded model answers (only with `--llm-model`) |

Consolidated memories keep `source_ids`, so compressed context is scored against the same
ground truth as raw context.

## Results

Output of `contextio run --users 100` (1,062 queries, seed 0). Costs assume placeholder prices of
\$3 / \$15 / \$0.30 per million input / output / cached tokens and 300 output tokens per request.

| strategy | budget | answerable | precision | stale rate | mean tokens | cache hit | mean cost |
|---|--:|--:|--:|--:|--:|--:|--:|
| full | none | 100.0% | 0.5% | 44.4% | 14,007 | 90.6% | $0.01228 |
| recency | 4000 | 10.6% | 0.4% | 8.1% | 3,999 | 90.2% | $0.00676 |
| bm25 | 1000 | 77.5% | 5.4% | 41.1% | 907 | 0.4% | $0.00726 |
| vector | 1000 | 73.7% | 4.6% | 41.3% | 998 | 0.1% | $0.00754 |
| vector | 8000 | 91.5% | 0.9% | 43.5% | 7,969 | 0.0% | $0.02844 |
| hybrid | 1000 | 84.0% | 4.5% | 42.0% | 998 | 0.3% | $0.00753 |
| hybrid | 4000 | 96.4% | 2.0% | 43.0% | 3,673 | 1.3% | $0.01543 |
| hybrid-knapsack | 500 | 92.6% | 4.5% | 42.2% | 497 | 9.2% | $0.00591 |
| hybrid-knapsack | 1000 | 97.6% | 3.4% | 43.0% | 978 | 5.6% | $0.00733 |
| compressed-full | none | 81.7% | 2.9% | 38.7% | 705 | 88.9% | $0.00493 |
| compressed-hybrid | 1000 | 79.2% | 6.2% | 37.3% | 312 | 23.4% | $0.00527 |

Fewest mean context tokens needed to reach a 90% answerable rate: **hybrid-knapsack 497**,
hybrid 3,673, vector 7,969, full 14,007. Recency, BM25, and both compressed strategies never
reach 90%.

What stands out:

1. **How you select matters as much as how you score.** Hybrid scoring with knapsack selection
   reaches 97.6% answerability at ~1k tokens, 14× less than full context, for a loss of 2.4 points.
   With the same scores, greedy top-k selection needs ~3.7k tokens to pass 90%. Knapsack fills the
   budget with short, dense memories (profile facts, one-line decisions) instead of long
   assistant replies.
2. **Recency is close to useless for long-lived users.** The facts that matter are spread across
   months, so "last N messages" covers only 10.6% of queries even at 4k tokens.
3. **Lexical and semantic retrieval fail on different queries.** At 1k tokens BM25 never finds
   style preferences (0%) and rarely finds background facts (33%), because the query shares no
   words with them. Pinning short static facts helps, but greedy hybrid still only reaches 17% on
   style queries because long, higher-scoring memories crowd out the pinned facts. Knapsack
   selection gets both categories to 100% (`contextio run --by-kind --budgets 1000`).
4. **Compression is the most token-efficient, but lossy.** Rule-based consolidation shrinks a
   user's ~14k tokens to ~700, and compressed-hybrid has the highest answerable rate per 1k
   tokens of any configuration (3.22 at a 250-token budget). It loses about 18 points because it
   misses facts phrased without cue words ("Redis it is for the caching layer").
5. **No strategy handles stale information.** Every strategy that reaches high recall also puts
   superseded facts in ~40% of contexts. Preference queries, where an old database preference
   competes with a newer one, are the hardest category (74% at best at 1k tokens). Retrieval
   cannot tell that a fact has been superseded; consolidation with conflict resolution is the
   open problem.
6. **Caching changes the cost picture.** Append-only full context keeps a stable prefix and gets
   a 90% cache hit rate, which cuts its cost from ~\$0.047 to ~\$0.012 per request. Selection still
   wins on cost, and on latency and attention dilution, which these numbers do not capture.

### Caveats

These numbers come from synthetic data, a heuristic tokenizer, and a hashing embedder that
stands in for a weak embedding model. "Answerable" means the evidence was in context, not that a
model used it correctly. Use `--llm-model` to measure that. Latency covers retrieval and
selection only, and the first query for each user includes building its index. Treat the results
as a way to compare strategies under controlled conditions, not as absolute numbers.

## Roadmap

- Real embedding models and rerankers behind the `Embedder` / `Retriever` protocols
- LLM consolidation that resolves conflicts so superseded facts are dropped, which targets the
  stale rate
- Learning hybrid weights from logged usage instead of hand-tuning them
- Ordering experiments (lost-in-the-middle) and context formatting
- Adapters for public long-term memory benchmarks
- Accuracy-vs-tokens plots

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

Contributions are welcome. Please open an issue to discuss larger changes.

## License

[MIT](LICENSE)
