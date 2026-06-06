#include "agent/cmd_exec.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
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

ExecResult execute_command(char *const argv[], int timeout_sec, char *out_buf, size_t out_buf_size) {
    ExecResult res = { .exit_code = -1, .timed_out = 0, .stdout_len = 0 };
    if (!argv || !argv[0]) return res;
    if (!exec_policy_allows(argv[0])) {
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
