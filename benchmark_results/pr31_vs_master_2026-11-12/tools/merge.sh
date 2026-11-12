echo "\$ date -u"; date -u; echo "\$ git merge --no-commit pr31   (in a worktree of origin/master fefa37e)"
echo "Automatic merge failed. Conflicts (src/, include/ => aborted, not hand-resolved):"; cat ~/evidence/merge_conflicts.txt
echo "\$ git merge --abort ; worktree removed.  => merge not clean: C5 (merged) not built"
