#!/usr/bin/env bash
# Build an orphan branch that holds only the judging materials (rubric, instructions, items),
# so judge sessions checked out on it cannot see run results or the condition key.
#   experiments/make_judge_branch.sh <items.jsonl> [branch]
set -euo pipefail
items=$1
branch=${2:-claude/e2v2-judging-items}
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
git -C "$repo" worktree add --detach "$tmp" >/dev/null
cd "$tmp"
git checkout --orphan "$branch" >/dev/null
git rm -rq --cached . && git clean -fdxq
cp "$repo/experiments/judging_v2/RUBRIC.md" "$repo/experiments/judging_v2/JUDGE_INSTRUCTIONS.md" .
cp "$items" items.jsonl
git add RUBRIC.md JUDGE_INSTRUCTIONS.md items.jsonl
git commit -qm "E2 v2 judging materials (rubric, instructions, $(wc -l < items.jsonl) blinded items)"
git push -q -u origin "$branch"
git -C "$repo" worktree remove --force "$tmp"
echo "pushed $branch"
