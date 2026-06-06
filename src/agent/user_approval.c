#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "agent/agent_loop.h"

int user_approval_prompt(const char *cmd) {
    if (!cmd) return 0;
    /* Auto-approve when env var PROJECT_ZERO_AGENT_AUTO_APPROVE is set to 1/true/Y/T */
    const char *autoenv = getenv("PROJECT_ZERO_AGENT_AUTO_APPROVE");
    if (autoenv &&
        (autoenv[0] == '1' || tolower((unsigned char)autoenv[0]) == 'y' || tolower((unsigned char)autoenv[0]) == 't')) {
        printf("\n\033[1;33m[AGENT] Auto-approving (env PROJECT_ZERO_AGENT_AUTO_APPROVE=1)\033[0m %s\n", cmd);
        return 1;
    }
    printf("\n\033[1;33m[AGENT] About to execute:\033[0m %s\n", cmd);
    printf("Allow? [y/N]: ");
    fflush(stdout);
    int c = getchar();
    /* Consume rest of line */
    while (c != '\n' && c != EOF) { int r = getchar(); if (r == EOF) break; if (r == '\n') break; }
    if (c == 'y' || c == 'Y') return 1;
    return 0;
}
