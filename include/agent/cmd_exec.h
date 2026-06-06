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

/* Quick allow-list check for a command name (argv[0]) */
int exec_policy_allows(const char *cmd_name);

/* Execute argv[] with a timeout (seconds). stdout (and stderr) are captured into out_buf.
 * Returns an ExecResult describing the run. */
ExecResult execute_command(char *const argv[], int timeout_sec, char *out_buf, size_t out_buf_size);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_CMD_EXEC_H */
