#include "agent/cmd_exec.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/select.h>

/* Very small allow-list to prevent destructive commands. Extend as needed. */
static const char *allowlist[] = {"echo", "ls", "cat", "pwd", "uname", "date", "id", NULL};

int exec_policy_allows(const char *cmd_name) {
    if (!cmd_name) return 0;
    for (size_t i = 0; allowlist[i] != NULL; i++) {
        if (strcmp(cmd_name, allowlist[i]) == 0) return 1;
    }
    return 0;
}

/* True if `arg` is a relative path with no ".." traversal segment anywhere
 * in it -- confines file-reading commands to the current working
 * directory's subtree. Rejects absolute paths (leading '/') and home-dir
 * shortcuts (leading '~'). A ".." must be its own path segment (bounded by
 * '/' or string start/end) to match -- "foo..bar" or "..." are not
 * traversal and are left alone. */
static int path_arg_is_safe(const char *arg) {
    if (!arg || !arg[0]) return 1;
    if (arg[0] == '/' || arg[0] == '~') return 0;
    size_t len = strlen(arg);
    for (size_t i = 0; i < len; i++) {
        if (arg[i] != '.') continue;
        int start_ok = (i == 0) || (arg[i - 1] == '/');
        if (!start_ok) continue;
        if (i + 1 < len && arg[i + 1] == '.') {
            int end_ok = (i + 2 == len) || (arg[i + 2] == '/');
            if (end_ok) return 0; /* found a ".." path segment */
        }
    }
    return 1;
}

int exec_policy_allows_args(char *const argv[]) {
    if (!argv || !argv[0]) return 0;
    if (!exec_policy_allows(argv[0])) return 0;

    if (strcmp(argv[0], "cat") == 0 || strcmp(argv[0], "ls") == 0) {
        for (int i = 1; argv[i]; i++) {
            /* Flags (leading '-') aren't paths; everything else is treated
             * as a path candidate and must stay within the CWD subtree. */
            if (argv[i][0] == '-') continue;
            if (!path_arg_is_safe(argv[i])) return 0;
        }
    } else if (strcmp(argv[0], "date") == 0) {
        /* `date` with no args, or a `+FORMAT` string, is read-only. Every
         * other form (notably -s/--set, which changes the system clock)
         * is rejected outright -- this allow-listed "diagnostic" command
         * has no business mutating host state. */
        for (int i = 1; argv[i]; i++) {
            if (argv[i][0] != '+') return 0;
        }
    }
    return 1;
}

ExecResult execute_command(char *const argv[], int timeout_sec, char *out_buf, size_t out_buf_size) {
    ExecResult res = { .exit_code = -1, .timed_out = 0, .stdout_len = 0 };
    if (!argv || !argv[0]) return res;
    if (!exec_policy_allows_args(argv)) {
        res.exit_code = 127; /* policy denied */
        return res;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return res;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return res;
    }

    if (pid == 0) {
        /* child */
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]); close(pipefd[1]);

        /* Defense-in-depth resource caps, independent of the parent's own
         * timeout/SIGKILL loop below (which is cooperative and could be
         * delayed by scheduler pressure): a hard CPU-time kill via SIGXCPU,
         * and a bounded address space. The allow-listed commands (echo, ls,
         * cat, pwd, uname, date, id) never legitimately need more than
         * this. Best-effort -- if setrlimit itself fails, still proceed to
         * exec rather than silently hang the agent turn. */
        struct rlimit cpu_limit;
        cpu_limit.rlim_cur = (rlim_t)(timeout_sec > 0 ? timeout_sec + 2 : 30);
        cpu_limit.rlim_max = cpu_limit.rlim_cur;
        setrlimit(RLIMIT_CPU, &cpu_limit);

        struct rlimit as_limit;
        as_limit.rlim_cur = (rlim_t)256 * 1024 * 1024; /* 256 MiB */
        as_limit.rlim_max = as_limit.rlim_cur;
        setrlimit(RLIMIT_AS, &as_limit);

        execvp(argv[0], argv);
        /* exec failed */
        _exit(127);
    }

    /* parent */
    close(pipefd[1]);

    /* Set non-blocking read */
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    size_t total_read = 0;
    int status = 0;
    time_t start = time(NULL);

    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(pipefd[0], &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; /* 100ms polling */
        int rv = select(pipefd[0] + 1, &rfds, NULL, NULL, &tv);
        if (rv > 0 && FD_ISSET(pipefd[0], &rfds)) {
            ssize_t r = read(pipefd[0], (out_buf ? out_buf + total_read : NULL), (out_buf_size > total_read ? out_buf_size - total_read - 1 : 0));
            if (r > 0) {
                total_read += (size_t)r;
                if (total_read >= out_buf_size - 1) break;
                continue;
            } else if (r == 0) {
                break; /* EOF */
            }
        }
        pid_t wp = waitpid(pid, &status, WNOHANG);
        if (wp == pid) break;
        if (timeout_sec > 0 && (time(NULL) - start) >= timeout_sec) {
            /* timeout */
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            res.timed_out = 1;
            break;
        }
    }

    close(pipefd[0]);
    if (out_buf && out_buf_size > 0) out_buf[total_read < out_buf_size ? total_read : out_buf_size - 1] = '\0';
    res.stdout_len = total_read;
    if (!res.timed_out) {
        if (WIFEXITED(status)) res.exit_code = WEXITSTATUS(status);
    }
    return res;
}
