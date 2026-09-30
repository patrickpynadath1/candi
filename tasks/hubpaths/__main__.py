"""python -m tasks.hubpaths {generate,diagnose,sample,sweep}."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from .config import HubPathsConfig, preset
from .data import build_dataset
from .diagnostics import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument(
        "--preset",
        default="tiny_debug",
        choices=[
            "default",
            "tiny_debug",
            "language_like",
            "sudoku_like",
            "flat_control",
        ],
    )
    generate.add_argument("--config", type=Path, help="JSON overrides")
    generate.add_argument(
        "--set", action="append", default=[], metavar="KEY=JSON_VALUE"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--mode", choices=["conditional", "latent_core"], default="conditional"
    )
    generate.add_argument(
        "--latent-config", type=Path, help="JSON latent-core overrides"
    )
    generate.add_argument(
        "--latent-set", action="append", default=[], metavar="KEY=JSON_VALUE"
    )
    analysis = sub.add_parser("diagnose")
    analysis.add_argument("dataset", type=Path)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument(
        "--split", default="train", choices=["train", "validation", "test", "test_ood"]
    )
    analysis.add_argument("--ngrams", default="1,2,3,4,5")
    analysis.add_argument("--max-examples", type=int)
    analysis.add_argument("--entropy-prompts", type=int, default=128)
    analysis.add_argument("--scope", default="path", choices=["path", "prompt", "full"])
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument(
        "--knob",
        choices=["gamma", "temperature", "backbone", "max_dist", "n_hints"],
        required=True,
    )
    sweep.add_argument("--values", required=True, help="JSON list, e.g. [2.1,2.5,3,4]")
    sweep.add_argument("--n-train", type=int, default=1000)
    sample = sub.add_parser(
        "sample", help="sample the ground-truth law for a supplied waypoint list"
    )
    sample.add_argument("dataset", type=Path)
    sample.add_argument("--waypoints", type=int, nargs="+", required=True)
    sample.add_argument("--count", type=int, default=1)
    sample.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "generate":
        values = preset(args.preset).to_dict()
        if args.config:
            values.update(json.loads(args.config.read_text()))
        for item in args.set:
            key, value = item.split("=", 1)
            values[key] = json.loads(value)
        if args.mode == "latent_core":
            from .latent import LatentCoreConfig, build_latent_dataset

            latent = (
                json.loads(args.latent_config.read_text()) if args.latent_config else {}
            )
            for item in args.latent_set:
                key, value = item.split("=", 1)
                latent[key] = json.loads(value)
            report = build_latent_dataset(
                HubPathsConfig(**values), LatentCoreConfig(**latent), args.output
            )
        else:
            if args.latent_config or args.latent_set:
                parser.error(
                    "--latent-config and --latent-set require --mode latent_core"
                )
            report = build_dataset(HubPathsConfig(**values), args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "diagnose":
        report = diagnose(
            args.dataset,
            args.output,
            args.split,
            tuple(map(int, args.ngrams.split(","))),
            args.max_examples,
            args.entropy_prompts,
            args.scope,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in [
                        "examples_analyzed",
                        "vocabulary_used",
                        "path_zipf_slope",
                        "path_unigram_entropy_bits",
                        "ngrams",
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "sample":
        import numpy as np
        from .graph import Graph, N_SPECIAL
        from .rsp import RSP
        from .routes import RouteDistribution

        metadata = json.loads((args.dataset / "metadata.json").read_text())
        if metadata.get("task_mode") in {"latent_core", "grammar_core"}:
            parser.error(
                "sample --waypoints is for the conditional task; latent-core generation uses hidden sidecars"
            )
        graph = Graph.load(args.dataset / "graph.npz")
        distribution = RouteDistribution(
            RSP(graph, cache_dir=args.dataset / "goal_cache"), args.waypoints
        )
        rng = np.random.default_rng(args.seed)
        paths = [distribution.sample(rng) for _ in range(args.count)]
        print(
            json.dumps(
                {
                    "waypoints": args.waypoints,
                    "conditional_entropy_bits": distribution.cond_entropy_bits(),
                    "paths": [
                        {
                            "nodes": path,
                            "true_logprob": distribution.true_logprob(
                                np.asarray(path) + N_SPECIAL
                            ),
                        }
                        for path in paths
                    ],
                },
                indent=2,
            )
        )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, value in enumerate(json.loads(args.values)):
            c = replace(
                preset("tiny_debug"),
                n_train=args.n_train,
                n_val=20,
                n_test=20,
                min_edge_count=0,
                **{args.knob: value},
            )
            destination = args.output / f"{index:02d}_{args.knob}_{value}"
            build_dataset(c, destination)
            report = diagnose(destination, entropy_prompts=32)
            rows.append(
                {
                    args.knob: value,
                    "zipf_slope": report["path_zipf_slope"],
                    "mean_path_length": report["path_length"]["mean"],
                    "vocabulary_used": report["vocabulary_used"],
                    "mean_conditional_entropy": report["conditional_entropy_bits"][
                        "mean"
                    ],
                }
            )
        (args.output / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (args.output / "sweep.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
