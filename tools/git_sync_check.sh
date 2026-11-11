#!/usr/bin/env bash
# git_sync_check.sh — report, at session start, whether this working copy is behind GitHub.
#
# Run by the Claude Code SessionStart hook (.claude/settings.json) and by hand
# (`tools/git_sync_check.sh`). Read-only apart from `git fetch`: it never pulls,
# merges or touches the working tree, so it is safe with uncommitted work.
# Prints nothing when the copy is current; otherwise one line per finding plus
# the command to fix it. Always exits 0 so a session never fails to start.
#
# 2026-11-11: added after a long-lived working copy on a server was found 168
# commits behind master (work landed from other sessions); see
# docs/ai/session-start-checklist.md step 0 and docs/ai/decision-log.md.

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

remote=origin
base=master
git remote get-url "$remote" >/dev/null 2>&1 || exit 0

if ! timeout 20 git fetch -q "$remote" 2>/dev/null; then
    echo "git sync: could not reach $remote; this copy may be stale (last fetched data shown below)."
fi

branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "(detached)")
dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

if git rev-parse --verify -q "refs/remotes/$remote/$base" >/dev/null &&
   git rev-parse --verify -q "refs/heads/$base" >/dev/null; then
    behind=$(git rev-list --count "$base..$remote/$base")
    if [ "$behind" -gt 0 ]; then
        if [ "$branch" = "$base" ] && [ "$dirty" -eq 0 ]; then
            echo "git sync: $base is $behind commit(s) behind $remote/$base. Run: git pull --ff-only $remote $base"
        else
            echo "git sync: local $base is $behind commit(s) behind $remote/$base. Run: git fetch $remote $base:$base (or switch to $base and git pull --ff-only)"
        fi
    fi
fi

if [ "$branch" != "$base" ] && [ "$branch" != "(detached)" ]; then
    if git rev-parse --verify -q "refs/remotes/$remote/$base" >/dev/null; then
        lag=$(git rev-list --count "HEAD..$remote/$base")
        [ "$lag" -gt 0 ] && echo "git sync: branch $branch is $lag commit(s) behind $remote/$base; rebase or merge before opening a PR."
    fi
    if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
        ahead=$(git rev-list --count "$upstream..HEAD")
        [ "$ahead" -gt 0 ] && echo "git sync: branch $branch has $ahead commit(s) not pushed to $upstream."
    else
        echo "git sync: branch $branch has no upstream on $remote; push it (git push -u $remote $branch) so the work is not only on this machine."
    fi
fi

[ "$dirty" -gt 0 ] && echo "git sync: $dirty uncommitted change(s) in the working tree."
exit 0
