#ifndef AGENT_CMD_EXEC_H
#define AGENT_CMD_EXEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int exit_code;
    int timed_out; /* 0/1 */
    size_t stdout_len;
} ExecResult;

/* Quick allow-list check for a command name (argv[0]) only -- does NOT
 * validate arguments. Kept for backward compatibility with existing callers/
 * tests; execute_command() itself always additionally calls
 * exec_policy_allows_args() below, which is the real gate. */
int exec_policy_allows(const char *cmd_name);

/*
 * Full policy check: command name AND its arguments. The name-only allow-
 * list (exec_policy_allows) is not enough on its own -- `cat`/`ls` can read
 * *any* file the process has permission to read (.env files, SSH keys,
 * /etc/shadow, anything else on the host), and a deployment with
 * PROJECT_ZERO_AGENT_AUTO_APPROVE=1 set gets zero human review of that
 * before it runs. This additionally confines `cat`/`ls` path arguments to
 * relative paths with no ".." traversal (i.e. the current working
 * directory's subtree only, no absolute paths, no escaping upward) and
 * restricts `date` to its read-only forms (no args, or a `+FORMAT` string
 * only -- blocks `-s`/`--set` and other state-mutating flags).
 * Returns 1 if the whole command (name + every argument) is allowed.
 */
int exec_policy_allows_args(char *const argv[]);

/* Execute argv[] with a timeout (seconds). stdout (and stderr) are captured into out_buf.
 * Returns an ExecResult describing the run. */
ExecResult execute_command(char *const argv[], int timeout_sec, char *out_buf, size_t out_buf_size);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_CMD_EXEC_H */
