"""Judge the TinyStories tokenizer-ladder generations with TypeSafe's Jev.

Reads the regrade archive (blog-runs-regrade-*/), never the GPU servers.

Tasks
  pick     Which continuation of a prompt is best (coherence + topicality).
           --group sc-controlled  the 3 diffusion methods at one vocab, all SC or all no SC (default)
           --group sc-pair        one method at one vocab, SC vs no SC
           --group method         one model across the 5 vocab sizes
           --group vs-ar          each diffusion model vs AR at the same vocab
           --group vocab          all 7 models at one vocab
           Every comparison is judged twice, with the options in opposite orders,
           and the two probability vectors are averaged to cancel position bias.
  vs-reference
           Model ending vs the real ending. Two questions: which is better, and
           which did the original author write (0.5 = indistinguishable).
  anchored Score coherence and topicality with the real ending shown only as a
           quality bar (far worse ... about as good ... better).
  garbled  For every continuation, one yes/no question per distinct word:
           is this word garbled? Gives a count and an expected count.
  rate     Absolute per-continuation ratings (scores + yes/no flags). The real
           endings are rated with the same questions, and the summary divides
           each model's scores by theirs.

usage
  export TYPESAFE_API_KEY=...
  uv run --no-project --python 3.12 --with typesafe-sdk python scripts/analysis/jev_ladder.py \
      ARCHIVE_DIR OUT_DIR pick --group vocab [--limit 20] [--dry-run]
  ... garbled | rate
  ... summarize          (reads OUT_DIR/*.jsonl, writes CSV tables)

Responses are appended to OUT_DIR/<task>[-<group>].jsonl keyed by request id, so
an interrupted run resumes where it stopped.
"""

import argparse
import asyncio
import csv
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

VOCABS = [260, 500, 1000, 5000, 10000]
METHODS = {
    "gaussian_embedding": "Gaussian",
    "masked": "Masked",
    "uniform_posterior_ce": "Uniform",
}
MODEL = "jev-1.14.0"  # pin the version so reruns are comparable


# --------------------------------------------------------------------------- data
def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def label_of(case):
    if "-ar-" in case:
        return "AR"
    method, arm = re.match(r"tinystories-bpe-\d+-(.+)-(none|expected_embedding)-seed", case).groups()
    return f"{METHODS[method]} {'SC' if arm == 'expected_embedding' else 'no SC'}"


def load_ladder(archive):
    """-> {(seed, example_index): {(vocab, label): record}} plus prompts/references."""
    rows = [r for r in read_json(f"{archive}/blog-cases.json") if r["figure"].startswith("I ")]
    table = defaultdict(dict)
    for r in rows:
        vocab = int(re.search(r"bpe-(\d+)-", r["case"]).group(1))
        for path in sorted(glob.glob(f"{archive}/{r['final_records_glob']}")):
            data = read_json(path)
            for rec in data["records"]:
                table[(data["seed"], int(rec["index"]))][(vocab, label_of(r["case"]))] = {
                    "case": r["case"], "prompt": rec["prompt"], "continuation": rec["continuation"],
                    "reference": rec["reference"], "eos": bool(rec.get("eos_present", True)),
                }
    return table


def story_cut(text, limit=1400):
    """Continuations end at a token budget; trim to the limit at a word boundary."""
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


# --------------------------------------------------------------------------- requests
PICK_INSTRUCTIONS = (
    "Below is the opening of a short children's story and several candidate continuations. "
    "Which continuation is the best next part of this story? Judge two things together. "
    "Coherence: the sentences make sense on their own and together, events follow from each other, "
    "and characters, objects and names stay consistent. "
    "Topicality: the continuation picks up the opening (finishes its last word or sentence sensibly) "
    "and stays on the story it sets up. "
    "Every continuation was cut off at a length limit, so ignore abrupt endings. "
    "Do not prefer a continuation for being longer, and do not reward stock phrases on their own."
)


def _choice_pair_requests(prompt, named, gname, seed, idx, rng, questions):
    """Two requests per comparison, options in opposite orders."""
    names = sorted(named)
    rng.shuffle(names)
    out = []
    for flip, perm in (("fwd", names), ("rev", names[::-1])):
        letters = [chr(65 + i) for i in range(len(perm))]
        state = f"Story opening:\n{prompt}\n\n" + "\n\n".join(
            f"Continuation {L}:\n{story_cut(named[n])}" for L, n in zip(letters, perm))
        out.append({
            "id": f"{seed}:{idx}:{gname}:{flip}", "seed": seed, "index": idx, "group": gname,
            "letters": dict(zip(letters, perm)), "state": state,
            "questions": {k: {"type": "choice", "instructions": ins,
                              "criteria": {L: f"Continuation {L}" for L in letters}} for k, ins in questions.items()},
        })
    return out


GROUPS = {
    # name: (description, function cell -> {group name: [(vocab, label), ...]})
    "vocab": ("all 7 models at one vocabulary size",
              lambda cell: {f"vocab={v}": [k for k in cell if k[0] == v] for v in VOCABS}),
    "sc-controlled": ("the 3 diffusion methods at one vocabulary size, all with SC or all without",
                      lambda cell: {f"vocab={v}|{arm}": [k for k in cell if k[0] == v and k[1].partition(" ")[2] == arm]
                                    for v in VOCABS for arm in ("SC", "no SC")}),
    "sc-pair": ("one method at one vocabulary size, with SC vs without",
                lambda cell: {f"vocab={v}|{m}": [(v, f"{m} SC"), (v, f"{m} no SC")]
                              for v in VOCABS for m in METHODS.values()}),
    "method": ("one model across the 5 vocabulary sizes",
               lambda cell: {f"model={lab}": [k for k in cell if k[1] == lab] for lab in sorted({k[1] for k in cell})}),
    "vs-ar": ("each diffusion model against AR at the same vocabulary size",
              lambda cell: {f"vocab={v}|{lab}": [(v, lab), (v, "AR")]
                            for (v, lab) in cell if lab != "AR"}),
}


def pick_requests(table, group, with_reference, rng):
    out = []
    for (seed, idx), cell in sorted(table.items()):
        first = next(iter(cell.values()))
        for gname, keys in GROUPS[group][1](cell).items():
            keys = [k for k in keys if k in cell]
            if len(keys) < 2:
                continue
            named = {(lab if group != "method" else f"vocab={v}"): cell[(v, lab)]["continuation"] for v, lab in keys}
            if with_reference:
                named["REFERENCE"] = first["reference"]
            out += _choice_pair_requests(first["prompt"], named, gname, seed, idx, rng, {"best": PICK_INSTRUCTIONS})
    return out


VS_REF = {
    "best": PICK_INSTRUCTIONS,
    "original": ("One of these two continuations was written by the original author of the story; the other was "
                 "written by a computer model. Which one did the original author write?"),
}


def vs_reference_requests(table, rng):
    """Model ending vs the real ending: which is better, and which is the original."""
    out = []
    for (seed, idx), cell in sorted(table.items()):
        for (v, lab), rec in sorted(cell.items()):
            named = {lab: rec["continuation"], "REFERENCE": rec["reference"]}
            out += _choice_pair_requests(rec["prompt"], named, f"vocab={v}|{lab}", seed, idx, rng, VS_REF)
    return out


ANCHORED = {
    "coherence": "Do the sentences and events of the candidate make sense together, with consistent characters and objects?",
    "topicality": "Does the candidate pick up the opening sensibly and stay on the story it sets up?",
}
ANCHOR_LEVELS = ["Far worse than the reference", "Clearly worse than the reference",
                 "Somewhat worse than the reference", "About as good as the reference", "Better than the reference"]


def anchored_requests(table):
    """Score a continuation with the real ending shown only as a quality bar."""
    out = []
    for (seed, idx), cell in sorted(table.items()):
        for (v, lab), rec in sorted(cell.items()):
            out.append({
                "id": f"{seed}:{idx}:{rec['case']}", "seed": seed, "index": idx, "vocab": v, "model": lab,
                "state": ("Story opening:\n" + rec["prompt"] +
                          "\n\nReference continuation, written by the original author. Use it only as a bar for "
                          "quality. The story could go many other ways; a candidate that tells a different story "
                          "is not worse for that, and resembling the reference earns nothing.\n" + story_cut(rec["reference"]) +
                          "\n\nCandidate continuation (cut off at a length limit; ignore the abrupt ending):\n" +
                          story_cut(rec["continuation"])),
                "questions": {k: {"type": "score", "instructions": f"Compared with the reference in quality only: {q}",
                                  "criteria": ANCHOR_LEVELS} for k, q in ANCHORED.items()},
            })
    return out


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words_of(prompt, continuation, eos):
    """Words of the continuation, joining a word split across the prompt boundary."""
    text = continuation
    tail = re.search(r"[A-Za-z']+$", prompt)
    if tail and text[:1].isalpha():
        text = tail.group(0) + text
    found = WORD.findall(text)
    if not eos and found and text.rstrip()[-1:].isalpha():
        found = found[:-1]  # last word was cut by the length limit
    return found


GARBLED = (
    'In the story text above, is the word "{w}" garbled? Garbled means a misspelled word, a made-up '
    "non-word, a broken fragment of a word, or two words fused together. A real English word that is "
    "used awkwardly or ungrammatically is not garbled. Names of characters are not garbled."
)


GARBLED_THRESHOLD = 0.8  # real words land near 0.5-0.75, clear garbles at 0.85+


def garbled_requests(table):
    out, seen = [], set()
    for (seed, idx), cell in sorted(table.items()):
        for (v, lab), rec in sorted(cell.items()):
            ws = words_of(rec["prompt"], rec["continuation"], rec["eos"])
            distinct = sorted({w for w in ws}, key=ws.index)
            if not distinct or (seed, idx, rec["case"]) in seen:
                continue
            seen.add((seed, idx, rec["case"]))
            out.append({
                "id": f"{seed}:{idx}:{rec['case']}", "seed": seed, "index": idx, "vocab": v, "model": lab,
                "n_words": len(ws), "counts": {w: ws.count(w) for w in distinct},
                "word_keys": {f"w{i}": w for i, w in enumerate(distinct)},
                "state": "Story text to check (the opening and the continuation joined):\n"
                         + story_cut(rec["prompt"] + rec["continuation"], 1500),
                "questions": {f"w{i}": {"type": "noul", "instructions": GARBLED.format(w=w)} for i, w in enumerate(distinct)},
            })
    return out


RATE = {
    "coherence": ("score", "How coherent is the story text? Do the sentences and events make sense together?",
                  ["Incoherent: sentences do not connect", "Mostly incoherent", "Partly coherent",
                   "Mostly coherent with small lapses", "Fully coherent"]),
    "topicality": ("score", "How well does the story text follow from the opening and stay on its story?",
                   ["Ignores the opening", "Loosely related", "Follows the opening but drifts", "Stays on topic"]),
    "fluency": ("score", "How grammatical and fluent is the English, ignoring the content?",
                ["Broken", "Many errors", "Some errors", "Fluent"]),
    "name_change": ("noul", "Does a character's name or identity change partway through without reason?"),
    "contradiction": ("noul", "Does the text contradict itself (for example an object is lost and also in hand)?"),
    "repetition_loop": ("noul", "Does the text repeat the same phrase, sentence or action over and over?"),
    "plausible_tinystories": ("noul", "Could this text plausibly come from a well-written children's story collection?"),
}


def rate_requests(table):
    out = []
    for (seed, idx), cell in sorted(table.items()):
        for (v, lab), rec in sorted(cell.items()):
            qs = {}
            for k, spec in RATE.items():
                qs[k] = {"type": spec[0], "instructions": spec[1]}
                if spec[0] == "score":
                    qs[k]["criteria"] = spec[2]
            out.append({
                "id": f"{seed}:{idx}:{rec['case']}", "seed": seed, "index": idx, "vocab": v, "model": lab,
                "state": "Story opening:\n" + rec["prompt"] + "\n\nStory text (cut off at a length limit; "
                         "ignore the abrupt ending):\n" + story_cut(rec["continuation"]),
                "questions": qs,
            })
        first = next(iter(cell.values()))
        out.append({**out[-1], "id": f"{seed}:{idx}:REFERENCE", "vocab": 0, "model": "REFERENCE",
                    "state": out[-1]["state"].rsplit("ending):\n", 1)[0] + "ending):\n" + story_cut(first["reference"])})
    return out


# --------------------------------------------------------------------------- calling Jev
async def run_requests(reqs, path, concurrency):
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        Noul,
        RetryPolicy,
        Score,
        TypeSafeAPIError,
    )

    build = {"choice": Choice, "noul": Noul, "score": Score}
    done = set()
    if os.path.exists(path):
        done = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to send -> {path}", file=sys.stderr)
    sem, lock, tokens = asyncio.Semaphore(concurrency), asyncio.Lock(), [0]
    retry = RetryPolicy(max_retries=6, backoff_initial=1.0, backoff_max=30.0)
    async with AsyncTypeSafeClient(model=MODEL, retry=retry) as client:
        with open(path, "a") as f:  # noqa: ASYNC230
            async def one(r):
                async with sem:
                    qs = {k: build[q["type"]](**{a: b for a, b in q.items() if a != "type"}) for k, q in r["questions"].items()}
                    try:
                        resp = await client.system_one(state=r["state"], questions=qs)
                    except TypeSafeAPIError as e:
                        print(f"FAIL {r['id']}: {e.status} {e.request_id}", file=sys.stderr)
                        return
                    rec = {k: v for k, v in r.items() if k not in ("state", "questions")}
                    rec["model_version"] = resp.model
                    rec["request_id"] = resp.request_id
                    rec["answers"] = {k: a.model_dump() for k, a in resp.answers.items()}
                    async with lock:
                        tokens[0] += resp.usage.input_tokens or 0
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
            await asyncio.gather(*(one(r) for r in todo))
    print(f"input tokens this run: {tokens[0]:,}", file=sys.stderr)


# --------------------------------------------------------------------------- summaries
def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)


def summarize(out):
    for path in sorted(glob.glob(f"{out}/pick-*.jsonl")) + glob.glob(f"{out}/vs-reference.jsonl"):
        share, wins, n = defaultdict(float), defaultdict(float), defaultdict(int)
        pair = defaultdict(dict)
        for r in read_jsonl(path):
            for q, a in r["answers"].items():
                key = (q, r["seed"], r["index"], r["group"])
                pair[key][r["id"].rsplit(":", 1)[1]] = {r["letters"][L]: p for L, p in a["probabilities"].items()}
        for (q, seed, idx, group), sides in pair.items():
            if len(sides) < 2:
                continue  # need both orders
            avg = {m: (sides["fwd"][m] + sides["rev"][m]) / 2 for m in sides["fwd"]}
            best = max(avg, key=avg.get)
            for m, p in avg.items():
                share[(q, group, m)] += p
                n[(q, group, m)] += 1
                wins[(q, group, m)] += m == best
        rows = [{"question": q, "group": g, "option": m, "prompts": n[(q, g, m)],
                 "mean_win_probability": round(share[(q, g, m)] / n[(q, g, m)], 4),
                 "win_rate": round(wins[(q, g, m)] / n[(q, g, m)], 4)} for (q, g, m) in sorted(n)]
        write_csv(path.replace(".jsonl", "-summary.csv"), rows)
    path = f"{out}/anchored.jsonl"
    if os.path.exists(path):
        agg = defaultdict(lambda: defaultdict(list))
        for r in read_jsonl(path):
            for k, a in r["answers"].items():
                agg[(r["vocab"], r["model"])][k].append(a["score"])
        rows = [{"vocab": v, "model": m, "samples": len(next(iter(d.values()))),
                 **{f"{k} (0=far worse, 3=as good, 4=better)": round(sum(x) / len(x), 4) for k, x in d.items()}}
                for (v, m), d in sorted(agg.items())]
        write_csv(f"{out}/anchored-summary.csv", rows)
    path = f"{out}/garbled.jsonl"
    if os.path.exists(path):
        agg = defaultdict(lambda: [0, 0.0, 0, 0])  # hard count, expected count, words, samples
        for r in read_jsonl(path):
            hard = exp = 0.0
            for k, a in r["answers"].items():
                c = r["counts"][r["word_keys"][k]]
                exp += c * a["noul"]
                hard += c * (a["noul"] > GARBLED_THRESHOLD)
            s = agg[(r["vocab"], r["model"])]
            s[0] += hard; s[1] += exp; s[2] += r["n_words"]; s[3] += 1
        rows = [{"vocab": v, "model": m, "samples": s[3],
                 f"garbled_per_100_words (p>{GARBLED_THRESHOLD})": round(100 * s[0] / s[2], 3),
                 "expected_garbled_per_100_words": round(100 * s[1] / s[2], 3),
                 "garbled_per_sample": round(s[0] / s[3], 3)} for (v, m), s in sorted(agg.items())]
        write_csv(f"{out}/garbled-summary.csv", rows)
    path = f"{out}/rate.jsonl"
    if os.path.exists(path):
        agg = defaultdict(lambda: defaultdict(list))
        for r in read_jsonl(path):
            for k, a in r["answers"].items():
                agg[(r["vocab"], r["model"])][k].append(a["score"] if a["type"] == "score" else a["noul"])
        mean = {key: {k: sum(x) / len(x) for k, x in d.items()} for key, d in agg.items()}
        ref = mean.get((0, "REFERENCE"), {})
        rows = []
        for (v, m), d in sorted(mean.items()):
            row = {"vocab": v or "", "model": m, "samples": len(next(iter(agg[(v, m)].values())))}
            for k, x in d.items():
                row[k] = round(x, 4)
                if RATE[k][0] == "score" and ref.get(k):
                    row[f"{k} / reference"] = round(x / ref[k], 4)
            rows.append(row)
        write_csv(f"{out}/rate-summary.csv", rows)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("task", choices=["pick", "vs-reference", "anchored", "garbled", "rate", "summarize"])
    ap.add_argument("--group", choices=list(GROUPS), default="sc-controlled",
                    help="; ".join(f"{k}: {v[0]}" for k, v in GROUPS.items()))
    ap.add_argument("--with-reference", action="store_true", help="add the real continuation as an option (sanity check)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (per seed order)")
    ap.add_argument("--seeds", default="1041,1042")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="build requests, print one and a size estimate, send nothing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.task == "summarize":
        return summarize(a.out)

    table = load_ladder(a.archive)
    seeds = {int(s) for s in a.seeds.split(",")}
    keys = sorted(k for k in table if k[0] in seeds)
    if a.limit:
        keys = keys[: a.limit]
    table = {k: table[k] for k in keys}
    rng = random.Random(0)
    if a.task == "pick":
        reqs = pick_requests(table, a.group, a.with_reference, rng)
        name = f"pick-{a.group}{'-ref' if a.with_reference else ''}"
    elif a.task == "vs-reference":
        reqs, name = vs_reference_requests(table, rng), "vs-reference"
    elif a.task == "anchored":
        reqs, name = anchored_requests(table), "anchored"
    elif a.task == "garbled":
        reqs, name = garbled_requests(table), "garbled"
    else:
        reqs, name = rate_requests(table), "rate"

    chars = sum(len(r["state"]) + len(json.dumps(r["questions"])) for r in reqs)
    print(f"{len(reqs)} requests, ~{chars / 4 / 1e6:.1f}M input tokens "
          f"(~${chars / 4 / 1e6 * 0.042:.2f} at $0.042/Mtok if billed once per request)", file=sys.stderr)
    if a.dry_run:
        r = reqs[0]
        print(json.dumps({k: v for k, v in r.items() if k != "questions"}, indent=1)[:3000])
        print(json.dumps(dict(list(r["questions"].items())[:3]), indent=1))
        return
    asyncio.run(run_requests(reqs, f"{a.out}/{name}.jsonl", a.concurrency))


if __name__ == "__main__":
    main()
