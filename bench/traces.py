"""Routing-trace slices and the deterministic batch sampler used by the benchmark.

A trace slice (data/traces/<model>/eval.txt) holds the top-k expert ids of real
decode tokens.  A benchmark batch draws `tokens_per_rank` rows per rank from the
slice with replacement, from one seeded RNG in (rank, token) order, so a batch is
a pure function of (model, ranks, ranks_per_node, budget_mib).

Budget semantics: `budget_mib` is the pre-top-k send budget per rank.  With a
`chunk_bytes` token row, tokens_per_rank = round(budget_mib*2^20 / (chunk*topk))
rounded to a multiple of topk (at least topk).
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "traces")


@dataclass(frozen=True)
class TraceSlice:
    model: str
    trace_model: str
    num_experts: int
    topk: int
    pool_sha: str
    eval_rows: tuple  # tuple[tuple[int, ...]]
    pool_rows: tuple


def _read_rows(path: str) -> tuple:
    with open(path) as f:
        return tuple(tuple(int(x) for x in line.split()) for line in f if line.strip())


def load_slice(model: str, data_dir: str = DATA_DIR) -> TraceSlice:
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    info = manifest["models"][model]
    mdir = os.path.join(data_dir, model)
    eval_rows = _read_rows(os.path.join(mdir, "eval.txt"))
    pool_rows = _read_rows(os.path.join(mdir, "pool.txt"))
    assert len(eval_rows) == info["rows"], (model, len(eval_rows), info["rows"])
    return TraceSlice(model, info["trace_model"], info["num_experts"], manifest["topk"],
                      info["pool_sha"], eval_rows, pool_rows)


def tokens_per_rank(budget_mib: float, chunk_bytes: int, topk: int) -> int:
    budget_bytes = budget_mib * (1 << 20)
    return max(topk, round(budget_bytes / (chunk_bytes * topk)) * topk)


def _fnv1a(s: str) -> int:
    h = 2166136261
    for c in s.encode():
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def batch_seed(ts: TraceSlice, ranks: int, ranks_per_node: int, budget_mib: float,
               chunk_bytes: int, instance: str = "001") -> int:
    """Seed of the batch sampler.  The key string is kept verbatim from the
    trace-generation tool that produced the published numbers, so batches match."""
    dataset_sha = hashlib.sha256(f"livecodebench/execution {ts.pool_sha}".encode()).hexdigest()[:12]
    params = {"dslots": "64:32", "layer": 5, "model": ts.trace_model, "pool": "decode",
              "pools": "livecodebench/execution", "poolsha": dataset_sha, "sem": "homog"}
    param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    b = int(budget_mib) if float(budget_mib).is_integer() else budget_mib
    key = f"trace|{param_str}|w{ranks}x{ranks_per_node}|b{b}|k{ts.topk}|c{chunk_bytes}|id{instance}"
    return _fnv1a(key)


def sample_batch(ts: TraceSlice, ranks: int, ranks_per_node: int, budget_mib: float,
                 chunk_bytes: int) -> list:
    """[ranks * tokens_per_rank] rows of top-k ids; token t belongs to rank t // T."""
    T = tokens_per_rank(budget_mib, chunk_bytes, ts.topk)
    rng = random.Random(batch_seed(ts, ranks, ranks_per_node, budget_mib, chunk_bytes))
    pool = ts.eval_rows
    n = len(pool)
    return [pool[rng.randrange(n)] for _ in range(ranks) for _ in range(T)]


def routing_text(rows: list, topk: int, num_experts: int) -> str:
    lines = [f"{len(rows)} {topk} {num_experts}"]
    lines += [" ".join(str(e) for e in r) for r in rows]
    return "\n".join(lines) + "\n"


# sha256 of routing_text(sample_batch(...)) for the published 4-node cells.
_REFERENCE = {
    ("qwen3", 16, 4, 1, 8192): "b64f221a2cefb1866e7c4d145cb2c1b312a0beaa22fc69eea680e994804109d6",
    ("qwen3", 16, 4, 4, 8192): "94b50a3099f4cbf4bd6cbd09cad7070076f2fd065ee168b3594de65641a0adc9",
    ("qwen3", 16, 4, 16, 8192): "073177b88e340c4ca722127f4ddaa5726d3fa778305e56b372853a7c8a6d626d",
    ("k2", 16, 4, 1, 14336): "5a8889ba1045d725fa6fe6a2e5ea244be4fc36e49183a65a4b3d883866389f83",
    ("k2", 16, 4, 4, 14336): "75bdb55b878618be26c47c22c20d1ff68146fca5f40b5ce2e1ffeda21be9e729",
    ("k2", 16, 4, 16, 14336): "b26191a8603f938dfe9a976f2052ad310397987e18ea312ad60d7814af1a1493",
}


def self_test() -> int:
    bad = 0
    for (model, W, L, b, chunk), want in _REFERENCE.items():
        ts = load_slice(model)
        rows = sample_batch(ts, W, L, b, chunk)
        got = hashlib.sha256(routing_text(rows, ts.topk, ts.num_experts).encode()).hexdigest()
        ok = got == want
        bad += not ok
        print(f"{model:5s} ranks={W} b{b:<2d} tokens/rank={tokens_per_rank(b, chunk, ts.topk):4d} "
              f"{'ok' if ok else 'MISMATCH ' + got[:12]}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="regenerate the published 4-node batches and compare hashes")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    ap.print_help()
