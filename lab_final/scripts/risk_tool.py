#!/usr/bin/env python3

"""
Smart Lock Risk Tool (DREAD-based)

What this script does
1) Loads the model (YAML or JSON). If not found at data/model.yaml, it prompts for a path.
2) Builds a DREAD-based risk ranking:
   - DREAD_avg = mean(Damage, Reproducibility, Exploitability, Affected users, Discoverability)
   - risk = DREAD_avg * likelihood
3) Runs a 10,000-trial Monte Carlo over the attack tree (supports AND/OR).
4) Generates a left→right DOT and PNG with p_mc (%) for nodes/leaves.
5) Saves:
   - outputs/risk_ranking.csv
   - outputs/monte_carlo_summary.json
   - outputs/attack_tree_mc.dot, outputs/attack_tree_mc.png
   - outputs/attack_tree_mc_probs.csv
   - data/model_with_mc.yaml (augmented with p_mc)
"""

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import os
from pathlib import Path
from typing import Dict, Any
import yaml

TRIALS = 10_000
SEED = 1337  # set to None for non-reproducible runs


def load_model(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    ext = path.suffix.lower()
    with path.open(encoding="utf-8") as f:
        if ext in (".json",):
            return json.load(f)
        return yaml.safe_load(f)


def prompt_for_model_path(default_hint: str) -> Path:
    print(f"[!] Could not find model at: {default_hint}")
    while True:
        user_in = input("Enter path to your model file (.yaml/.yml/.json), or press Enter to quit: ").strip()
        if not user_in:
            raise SystemExit("No model provided. Exiting.")
        expanded = os.path.expandvars(user_in)
        p = Path(expanded).expanduser()
        if p.exists():
            return p
        print(f"[!] Not found: {p}. Please try again.")


def get_tree(model: Dict[str, Any]) -> Dict[str, Any]:
    tree = model.get("tree") or model.get("attack_tree")
    if tree is None:
        raise SystemExit("Model must contain a 'tree' (or 'attack_tree') section.")
    return tree


def threats_by_id(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for t in model.get("threats", []):
        result[t["id"]] = t
    return result


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False lets non-ASCII stay readable; safe on UTF-8 files
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def esc(s: str) -> str:
    return str(s).replace('"', '\\"')


# DREAD-based ranking

def dread_avg(t: Dict[str, Any]) -> float:
    d = t.get("dread")
    if isinstance(d, dict):
        keys = ("damage", "reproducibility", "exploitability", "affected_users", "discoverability")
        vals = [float(d.get(k, 0.0)) for k in keys]
        return sum(vals) / len(vals) if vals else 0.0
    return float(t.get("impact", 0.0))


def compute_risk_ranking_dread(threats: Dict[str, Dict[str, Any]]):
    rows = []
    for t in threats.values():
        p = float(t.get("likelihood", 0.0))
        d_avg = dread_avg(t)
        rows.append({
            "id": t["id"],
            "title": t.get("title", t["id"]),
            "affects": "|".join(t.get("affects", [])),
            "likelihood": p,
            "dread_avg": d_avg,
            "risk": d_avg * p,  # 0–10 times 0–1
        })
    rows.sort(key=lambda r: r["risk"], reverse=True)
    return rows


# Monte Carlo engine
def _node_val(node_id, nodes, leaves, leaf_truth, memo):
    if node_id in memo:
        return memo[node_id]
    if node_id in leaves:
        value = leaf_truth[node_id]
        memo[node_id] = value
        return value

    n = nodes[node_id]
    node_type = str(n.get("type", "OR")).upper()
    child_ids = n.get("children", [])
    child_values = [_node_val(cid, nodes, leaves, leaf_truth, memo) for cid in child_ids]

    if node_type == "AND":
        value = all(child_values) if child_values else False
    else:
        value = any(child_values)
    memo[node_id] = value
    return value


def mc_node_leaf_probs(model, threats, trials=TRIALS, seed=SEED):
    tree = get_tree(model)
    nodes = {n["id"]: n for n in tree["nodes"]}
    leaves = {l["id"]: l for l in tree["leaves"]}

    rng = random.Random(seed) if seed is not None else random.Random()

    count = {**{nid: 0 for nid in nodes}, **{lid: 0 for lid in leaves}}

    for _ in range(trials):
        leaf_truth = {}
        for lid, leaf in leaves.items():
            threat_id = leaf["threat_id"]
            p = float(threats[threat_id].get("likelihood", 0.0))
            leaf_truth[lid] = (rng.random() < p)

        memo = {}
        # Evaluate all nodes
        for nid in nodes:
            _node_val(nid, nodes, leaves, leaf_truth, memo)

        # FIXED counting: count leaves once, nodes once
        for lid, truth in leaf_truth.items():
            if truth:
                count[lid] += 1
        for nid in nodes:
            if memo.get(nid, False):
                count[nid] += 1

    probs = {k: c / trials for k, c in count.items()}
    node_probs = {nid: probs[nid] for nid in nodes}
    leaf_probs = {lid: probs[lid] for lid in leaves}
    return {"node_probs": node_probs, "leaf_probs": leaf_probs}


def mc_root_summary(model, threats, trials=TRIALS, seed=SEED):
    tree = get_tree(model)
    nodes = {n["id"]: n for n in tree["nodes"]}
    leaves = {l["id"]: l for l in tree["leaves"]}
    root = tree["nodes"][0]["id"]

    rng = random.Random(seed) if seed is not None else random.Random()
    successes = 0

    for _ in range(trials):
        leaf_truth = {}
        for lid, leaf in leaves.items():
            p = float(threats[leaf["threat_id"]].get("likelihood", 0.0))
            leaf_truth[lid] = (rng.random() < p)

        memo = {}
        if _node_val(root, nodes, leaves, leaf_truth, memo):
            successes += 1

    p_mean = successes / trials
    se = math.sqrt(max(p_mean * (1 - p_mean) / trials, 1e-12))
    ci_low = max(0.0, p_mean - 1.96 * se)
    ci_high = min(1.0, p_mean + 1.96 * se)
    return {"root": root, "trials": trials, "successes": successes, "p_mean": p_mean, "p_ci95": [ci_low, ci_high]}


# DOT rendering
def write_mc_dot(model, threats, node_probs, leaf_probs, dot_out: Path):
    tree = get_tree(model)
    nodes = {n["id"]: n for n in tree["nodes"]}
    leaves = {l["id"]: l for l in tree["leaves"]}

    lines = []
    lines.append('digraph AttackTreeMC {')
    lines.append('  rankdir=LR;')
    lines.append('  labelloc=top;')
    lines.append('  label="Attack Tree (10k Monte Carlo)";')
    lines.append('  graph [ordering="out"];')
    lines.append('  nodesep=0.4; ranksep=0.6;')
    lines.append('  node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica"];')

    for nid, n in nodes.items():
        node_type = str(n.get("type", "OR")).upper()
        name = n.get("name", nid)
        p = node_probs[nid] * 100
        lines.append(f'  {nid} [label="{esc(nid)}: {esc(name)}\\n({node_type})  p_mc≈{p:.1f}%", tooltip="{esc(name)}"];')

    for lid, l in leaves.items():
        t = threats[l["threat_id"]]
        title = t.get("title", l["threat_id"])
        d_avg = dread_avg(t)
        p_in = float(t.get("likelihood", 0.0)) * 100
        p_mc = leaf_probs[lid] * 100
        lines.append(
            f'  {lid} [label="{esc(lid)}: {esc(title)}\\nD={d_avg:.1f}  p_in={p_in:.0f}%  p_mc≈{p_mc:.1f}%", shape=note, tooltip="{esc(title)}"];'
        )

    for n in tree["nodes"]:
        children = n.get("children", [])
        if children:
            lines.append(f'  {n["id"]} -> {{ {" ".join(children)} }};')

    lines.append('}')
    dot_out.parent.mkdir(parents=True, exist_ok=True)
    dot_out.write_text("\n".join(lines), encoding="utf-8")  # ← key change


def render_png(dot_path: Path, png_out: Path):
    if shutil.which("dot") is None:
        print("[mc] Graphviz not found → skipping PNG render.")
        return
    png_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_out)], check=True)
    print(f"[mc] wrote {png_out}")


def main():
    parser = argparse.ArgumentParser(description="Smart Lock Risk Tool (DREAD-based)")
    parser.add_argument("--model", type=str, help="Path to model file (.yaml/.yml/.json)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Resolve model path: CLI > default > prompt
    model_path = Path(args.model).expanduser() if args.model else (root / "data" / "model.yaml")
    if not model_path.exists():
        try:
            model_path = prompt_for_model_path(str(model_path))
        except SystemExit as e:
            print(e)
            return

    # Outputs
    out_rankcsv   = root / "outputs" / "risk_ranking.csv"
    out_mcjson    = root / "outputs" / "monte_carlo_summary.json"
    out_mcdot     = root / "outputs" / "attack_tree_mc.dot"
    out_mcpng     = root / "outputs" / "attack_tree_mc.png"
    out_mcprobcsv = root / "outputs" / "attack_tree_mc_probs.csv"
    out_augmodel  = root / "data"    / "model_with_mc.yaml"

    # Load model
    model = load_model(model_path)
    threats = threats_by_id(model)

    # DREAD-based ranking
    ranking_rows = compute_risk_ranking_dread(threats)
    write_csv(ranking_rows, out_rankcsv)

    # Monte Carlo
    mc_summary = mc_root_summary(model, threats, trials=TRIALS, seed=SEED)
    write_json(mc_summary, out_mcjson)

    probs = mc_node_leaf_probs(model, threats, trials=TRIALS, seed=SEED)
    node_p, leaf_p = probs["node_probs"], probs["leaf_probs"]

    # Augment model with p_mc
    tree_key = "tree" if "tree" in model else "attack_tree"
    aug_tree = {"nodes": [], "leaves": []}
    for n in model[tree_key]["nodes"]:
        new_node = dict(n)
        new_node["p_mc"] = float(node_p[n["id"]])
        aug_tree["nodes"].append(new_node)
    for l in model[tree_key]["leaves"]:
        new_leaf = dict(l)
        new_leaf["p_mc"] = float(leaf_p[l["id"]])
        aug_tree["leaves"].append(new_leaf)
    augmented = dict(model)
    augmented[tree_key] = aug_tree
    out_augmodel.parent.mkdir(parents=True, exist_ok=True)
    out_augmodel.write_text(yaml.safe_dump(augmented, sort_keys=False), encoding="utf-8")

    # Diagram (DOT + PNG)
    write_mc_dot(model, threats, node_p, leaf_p, out_mcdot)
    render_png(out_mcdot, out_mcpng)

    # Combined p_mc ranking (nodes + leaves)
    rows = []
    tree = get_tree(model)
    nodes = {n["id"]: n for n in tree["nodes"]}
    leaves = {l["id"]: l for l in tree["leaves"]}
    for nid, p in node_p.items():
        rows.append({"id": nid, "type": "node", "name": nodes[nid].get("name", nid),
                     "dread_avg": "", "p_input": "", "p_mc": p})
    for lid, p in leaf_p.items():
        t = threats[leaves[lid]["threat_id"]]
        rows.append({"id": lid, "type": "leaf", "name": t.get("title", lid),
                     "dread_avg": dread_avg(t), "p_input": t.get("likelihood", ""),
                     "p_mc": p})
    rows.sort(key=lambda r: r["p_mc"], reverse=True)
    write_csv(rows, out_mcprobcsv)

    # Console summary
    print(
        f"\nModel: {model_path}\n"
        f"MC root {mc_summary['root']}: p_mean={mc_summary['p_mean']:.4f}  "
        f"CI95=[{mc_summary['p_ci95'][0]:.4f}, {mc_summary['p_ci95'][1]:.4f}] "
        f"over {TRIALS} trials"
    )


if __name__ == "__main__":
    main()
