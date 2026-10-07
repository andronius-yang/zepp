"""Greedy token agreement between two servers (or a server and a saved run).

  python -m zepp_sglang.check_tokens run --url http://host:30000 --out a.json [--n 48 --max-new 64]
  python -m zepp_sglang.check_tokens compare a.json b.json

`run` sends N fixed prompts with temperature 0 and stores the output token ids; `compare` reports
the fraction of identical sequences and the mean agreeing prefix length. Compare a stock-SGLang run
with a zepp run; a second stock run gives the run-to-run agreement of bf16 serving to compare against.
"""
import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Explain why the sky is blue in two sentences.", "Write a Python function that reverses a linked list.",
    "What is the capital of Australia and what is it known for?", "Summarize the plot of Romeo and Juliet.",
    "List three differences between TCP and UDP.", "Translate 'good morning, how are you?' into French and German.",
    "What causes the seasons on Earth?", "Give me a recipe for a simple tomato soup.",
    "Prove that the square root of 2 is irrational.", "Describe how a hash table handles collisions.",
    "What were the main causes of World War I?", "Write a haiku about autumn rain.",
    "How does public-key cryptography work?", "Explain the difference between a virus and a bacterium.",
    "What is the time complexity of quicksort and why?", "Describe the water cycle for a ten-year-old.",
    "Solve: a train travels 120 km in 1.5 hours; what is its average speed?", "What is a mixture-of-experts model?",
    "Name four planets and one fact about each.", "How do vaccines train the immune system?",
    "Write a SQL query that finds the second highest salary.", "What is the Pythagorean theorem used for?",
    "Explain compound interest with a numeric example.", "Why do leaves change color in the fall?",
    "Describe three uses of machine learning in medicine.", "What is the difference between weather and climate?",
    "Write a short limerick about a cat and a laptop.", "How does a binary search work?",
    "What is photosynthesis?", "Give two arguments for and against nuclear power.",
    "Explain what a derivative measures in calculus.", "How is bread leavened?",
    "What does an operating system scheduler do?", "Describe the rules of chess for the knight.",
    "What is the greenhouse effect?", "Write a function to check whether a string is a palindrome.",
    "Who painted the Mona Lisa and when?", "What are the phases of mitosis?",
    "Explain how GPS determines a position.", "What is the difference between RAM and storage?",
    "Compose a two-line rhyme about coffee.", "How do airplanes stay in the air?",
    "What is inflation and what causes it?", "Describe the structure of DNA.",
    "How does recursion differ from iteration?", "What is the boiling point of water at high altitude and why?",
    "Explain the concept of supply and demand.", "What is a black hole?",
]


def generate(url, prompt, max_new):
    body = json.dumps({"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}).encode()
    req = urllib.request.Request(url + "/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=len(PROMPTS)); r.add_argument("--max-new", type=int, default=64)
    r.add_argument("--parallel", type=int, default=16)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        with ThreadPoolExecutor(a.parallel) as ex:
            gens = list(ex.map(lambda p: generate(a.url, p, a.max_new), PROMPTS[:a.n]))
        outs = [{"prompt": p, "output_ids": o["output_ids"], "text": o["text"]} for p, o in zip(PROMPTS[:a.n], gens)]
        json.dump(outs, open(a.out, "w"))
        print(f"saved {len(outs)} generations to {a.out}")
        return
    A, B = json.load(open(a.a)), json.load(open(a.b))
    assert len(A) == len(B)
    same = 0; prefix = []
    for x, y in zip(A, B):
        ia, ib = x["output_ids"], y["output_ids"]
        same += ia == ib
        k = 0
        while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
            k += 1
        prefix.append(k / max(len(ia), 1))
    print(f"identical sequences {same}/{len(A)} ({100 * same / len(A):.0f} %), mean agreeing prefix "
          f"{100 * sum(prefix) / len(prefix):.1f} % of the output")
    sys.exit(0)


if __name__ == "__main__":
    main()
