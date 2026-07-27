"""Arena Elo/Bradley-Terry leaderboard — computes ratings from pairwise match results."""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

__all__ = [
    "normalize_matches",
    "compute_elo",
    "compute_bradley_terry",
    "build_leaderboard",
    "summarize_arena",
    "format_leaderboard_text",
    "format_leaderboard_markdown",
    "format_leaderboard_json",
]


def normalize_matches(data):
    """
    Normalize match data from various input formats.

    Accepts:
    - List of dicts with "model_a", "model_b", "winner" keys
    - Dict with "model_a", "model_b", "results" keys (arena output)
    - List of such dicts (multiple arena runs)
    """
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict):
            if "model_a" in data[0] and "model_b" in data[0]:
                if "results" in data[0]:
                    # List of arena outputs
                    flat = []
                    for item in data:
                        flat.extend(_expand_arena_output(item))
                    return flat
                elif "winner" in data[0]:
                    # List of flat matches
                    return data
        return []
    elif isinstance(data, dict):
        # Single arena output
        if "model_a" in data and "model_b" in data and "results" in data:
            return _expand_arena_output(data)
    return []


def _expand_arena_output(item):
    """Expand a single arena output dict into flat matches."""
    matches = []
    for result in item.get("results", []):
        match = {
            "model_a": item["model_a"],
            "model_b": item["model_b"],
            "winner": result.get("winner"),
        }
        if "category" in result:
            match["category"] = result["category"]
        matches.append(match)
    return matches


def compute_elo(matches, *, k=32.0, initial_rating=1500.0):
    """
    Compute Elo ratings from match results.

    Standard Elo algorithm:
    - Start all models at initial_rating
    - Process matches in order
    - For win: winner gets +K*(1-expected), loser gets -K*expected
    - For tie: each gets K*(0.5-expected)
    - expected_a = 1 / (1 + 10^((rating_b - rating_a)/400))

    Args:
        matches: List of dicts with "model_a", "model_b", "winner" keys
        k: K-factor (rating volatility)
        initial_rating: Starting rating for all models

    Returns:
        Dict mapping model names to Elo ratings
    """
    # Initialize ratings for all models
    ratings = defaultdict(lambda: initial_rating)

    # Ensure all models are initialized
    for match in matches:
        if match["model_a"] not in ratings:
            ratings[match["model_a"]] = initial_rating
        if match["model_b"] not in ratings:
            ratings[match["model_b"]] = initial_rating

    # Process matches in order
    for match in matches:
        a = match["model_a"]
        b = match["model_b"]
        winner = match["winner"]

        rating_a = ratings[a]
        rating_b = ratings[b]

        # Calculate expected score
        expected_a = 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a

        # Update ratings based on result
        if winner == "model_a":
            ratings[a] = rating_a + k * (1.0 - expected_a)
            ratings[b] = rating_b + k * (0.0 - expected_b)
        elif winner == "model_b":
            ratings[a] = rating_a + k * (0.0 - expected_a)
            ratings[b] = rating_b + k * (1.0 - expected_b)
        elif winner == "tie":
            ratings[a] = rating_a + k * (0.5 - expected_a)
            ratings[b] = rating_b + k * (0.5 - expected_b)

    return dict(ratings)


def compute_bradley_terry(matches, *, max_iter=100, tol=1e-6):
    """
    Compute Bradley-Terry strength parameters from match results.

    Iterative Bradley-Terry MLE:
    - Initialize all model strengths to 1.0
    - Repeat until convergence (max_iter):
      - For each model i: strength_i = wins_i / sum_over_opponents_j(n_ij / (strength_i + strength_j))
        where n_ij = total games between i and j (ties count 0.5)
        wins_i = wins against others (ties count 0.5)
      - Normalize so sum of strengths = number of models
      - Check convergence: max |new - old| < tol

    Args:
        matches: List of dicts with "model_a", "model_b", "winner" keys
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Dict mapping model names to strength scores
    """
    if not matches:
        return {}

    # Get all models
    models = set()
    for match in matches:
        models.add(match["model_a"])
        models.add(match["model_b"])

    if len(models) <= 1:
        return {model: 1.0 for model in models}

    models = sorted(models)
    num_models = len(models)

    # Initialize strengths
    strengths = {model: 1.0 for model in models}

    # Count games and wins
    games = defaultdict(lambda: defaultdict(int))  # games[i][j] = num games between i and j
    wins = defaultdict(float)  # wins[i] = total wins by i (ties count as 0.5)

    for match in matches:
        a = match["model_a"]
        b = match["model_b"]
        winner = match["winner"]

        games[a][b] += 1
        games[b][a] += 1

        if winner == "model_a":
            wins[a] += 1.0
        elif winner == "model_b":
            wins[b] += 1.0
        else:  # tie
            wins[a] += 0.5
            wins[b] += 0.5

    # Iterative update
    for iteration in range(max_iter):
        new_strengths = {}

        for i in models:
            numerator = wins[i]
            denominator = 0.0

            for j in models:
                if i != j and games[i][j] > 0:
                    denominator += games[i][j] / (strengths[i] + strengths[j])

            if denominator > 0:
                new_strengths[i] = numerator / denominator
            else:
                new_strengths[i] = strengths[i]

        # Normalize so sum = num_models
        total = sum(new_strengths.values())
        for i in models:
            new_strengths[i] = new_strengths[i] * num_models / total

        # Check convergence
        max_change = max(abs(new_strengths[i] - strengths[i]) for i in models)
        if max_change < tol:
            break

        strengths = new_strengths

    return strengths


def build_leaderboard(matches, *, method="elo", k=32.0, initial_rating=1500.0):
    """
    Build a leaderboard from match results.

    Returns sorted list (descending by rating):
    [{
        "rank": int,          # 1-indexed
        "model": str,
        "rating": float,      # elo or bt score (rounded to 2 decimals)
        "wins": int,
        "losses": int,
        "ties": int,
        "total_games": int,
        "win_rate": float,    # wins / (wins+losses), None if 0 games
    }]

    Args:
        matches: List of match dicts
        method: "elo" or "bradley_terry"
        k: Elo K-factor
        initial_rating: Elo initial rating

    Returns:
        List of dicts with rank, model, rating, wins, losses, ties, total_games, win_rate
    """
    # Compute ratings
    if method == "elo":
        ratings = compute_elo(matches, k=k, initial_rating=initial_rating)
    elif method == "bradley_terry":
        ratings = compute_bradley_terry(matches)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Count wins/losses/ties
    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for match in matches:
        a = match["model_a"]
        b = match["model_b"]
        winner = match["winner"]

        if winner == "model_a":
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
        elif winner == "model_b":
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
        elif winner == "tie":
            stats[a]["ties"] += 1
            stats[b]["ties"] += 1

    # Build leaderboard
    leaderboard = []
    sorted_models = sorted(ratings.items(), key=lambda x: -x[1])

    for rank, (model, rating) in enumerate(sorted_models, 1):
        s = stats[model]
        total_games = s["wins"] + s["losses"] + s["ties"]
        win_rate = s["wins"] / (s["wins"] + s["losses"]) if (s["wins"] + s["losses"]) > 0 else None

        leaderboard.append({
            "rank": rank,
            "model": model,
            "rating": round(rating, 2),
            "wins": s["wins"],
            "losses": s["losses"],
            "ties": s["ties"],
            "total_games": total_games,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
        })

    return leaderboard


def summarize_arena(matches, *, method="elo"):
    """
    Summarize arena results with leaderboard, categories, and head-to-head.

    Returns:
    {
        "total_matches": int,
        "models": list[str],              # all unique models
        "method": str,                     # "elo" or "bradley_terry"
        "leaderboard": list[dict],         # from build_leaderboard
        "category_breakdown": dict,        # {category: {model: {wins, losses, ties}}}
        "head_to_head": dict,              # {f"{a}_vs_{b}": {"a_wins": int, "b_wins": int, "ties": int}}
    }

    Args:
        matches: List of match dicts
        method: "elo" or "bradley_terry"

    Returns:
        Dict with total_matches, models, method, leaderboard, category_breakdown, head_to_head
    """
    leaderboard = build_leaderboard(matches, method=method)

    # Get all models
    models = set()
    for match in matches:
        models.add(match["model_a"])
        models.add(match["model_b"])
    models = sorted(models)

    # Category breakdown
    category_breakdown = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    for match in matches:
        cat = match.get("category", "overall")
        a = match["model_a"]
        b = match["model_b"]
        winner = match["winner"]

        if winner == "model_a":
            category_breakdown[cat][a]["wins"] += 1
            category_breakdown[cat][b]["losses"] += 1
        elif winner == "model_b":
            category_breakdown[cat][b]["wins"] += 1
            category_breakdown[cat][a]["losses"] += 1
        elif winner == "tie":
            category_breakdown[cat][a]["ties"] += 1
            category_breakdown[cat][b]["ties"] += 1

    # Convert defaultdicts to regular dicts
    category_breakdown = {cat: dict(models_dict) for cat, models_dict in category_breakdown.items()}
    for cat in category_breakdown:
        for model in category_breakdown[cat]:
            category_breakdown[cat][model] = dict(category_breakdown[cat][model])

    # Head to head
    h2h = {}
    for match in matches:
        a = match["model_a"]
        b = match["model_b"]
        # Use alphabetical order for key
        key = "_vs_".join(sorted([a, b]))

        if key not in h2h:
            h2h[key] = {"a_wins": 0, "b_wins": 0, "ties": 0}

        # Determine which model is first in the sorted key
        models_sorted = sorted([a, b])
        first = models_sorted[0]

        winner = match["winner"]
        if winner == "model_a":
            if a == first:
                h2h[key]["a_wins"] += 1
            else:
                h2h[key]["b_wins"] += 1
        elif winner == "model_b":
            if b == first:
                h2h[key]["a_wins"] += 1
            else:
                h2h[key]["b_wins"] += 1
        elif winner == "tie":
            h2h[key]["ties"] += 1

    return {
        "total_matches": len(matches),
        "models": models,
        "method": method,
        "leaderboard": leaderboard,
        "category_breakdown": category_breakdown,
        "head_to_head": h2h,
    }


def format_leaderboard_text(summary):
    """Format leaderboard as plain text table."""
    lines = [f"=== ARENA LEADERBOARD ({summary['method'].upper()}) ==="]
    lines.append(f"{'Rank':<5} {'Model':<20} {'Rating':<10} {'W':<5} {'L':<5} {'T':<5} {'WinRate':<10}")

    for entry in summary["leaderboard"]:
        wr = f"{entry['win_rate']*100:.1f}%" if entry['win_rate'] is not None else "N/A"
        line = (
            f"{entry['rank']:<5} {entry['model']:<20} {entry['rating']:<10.2f} "
            f"{entry['wins']:<5} {entry['losses']:<5} {entry['ties']:<5} {wr:<10}"
        )
        lines.append(line)

    lines.append(f"\nTotal matches: {summary['total_matches']}")
    return "\n".join(lines)


def format_leaderboard_markdown(summary):
    """Format leaderboard as markdown table."""
    lines = [f"# ARENA LEADERBOARD ({summary['method'].upper()})"]
    lines.append("")
    lines.append("| Rank | Model | Rating | W | L | T | Win Rate |")
    lines.append("|------|-------|--------|---|---|---|----------|")

    for entry in summary["leaderboard"]:
        wr = f"{entry['win_rate']*100:.1f}%" if entry['win_rate'] is not None else "N/A"
        line = (
            f"| {entry['rank']} | {entry['model']} | {entry['rating']:.2f} | "
            f"{entry['wins']} | {entry['losses']} | {entry['ties']} | {wr} |"
        )
        lines.append(line)

    lines.append("")
    lines.append(f"**Total matches:** {summary['total_matches']}")
    return "\n".join(lines)


def format_leaderboard_json(summary):
    """Format leaderboard as JSON."""
    return json.dumps(summary, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Arena Elo/Bradley-Terry leaderboard")
    parser.add_argument("results", help="Arena results JSON file (run_arena.py output or flat match list)")
    parser.add_argument("--method", choices=["elo", "bradley_terry"], default="elo", help="Rating method")
    parser.add_argument("--k", type=float, default=32.0, help="Elo K factor")
    parser.add_argument("--initial-rating", type=float, default=1500.0, help="Elo initial rating")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--category", help="Filter to specific category")

    args = parser.parse_args()

    # Load results
    try:
        with open(args.results) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {args.results}: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize matches
    matches = normalize_matches(data)

    # Filter by category if requested
    if args.category:
        matches = [m for m in matches if m.get("category") == args.category]

    if not matches:
        print("No matches found", file=sys.stderr)
        sys.exit(1)

    # Summarize
    summary = summarize_arena(matches, method=args.method)

    # Format output
    if args.format == "text":
        output = format_leaderboard_text(summary)
    elif args.format == "markdown":
        output = format_leaderboard_markdown(summary)
    else:  # json
        output = format_leaderboard_json(summary)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
