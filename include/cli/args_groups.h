#ifndef TN_CLI_ARGS_GROUPS_H
#define TN_CLI_ARGS_GROUPS_H

/* The flag groups parse_args() tries in turn (src/cli/args_flags_*.c). Each takes the argument at
 * argv[*ip]: when it is one of the group's flags (with its value present) the group parses it,
 * advances *ip past any value it consumed, sets *handled and returns TN_OK; when it is not, nothing
 * changes and *handled stays 0; a bad value prints its message and returns TN_ERR_INVALID_CONFIG. */

#include "cli/args.h"

/* model and prompt inputs, sampling, penalties, stop strings and the watermark. */
TernaryError args_parse_generation(CliArgs *args, int argc, char **argv, int *ip, int *handled);

/* perplexity, RoPE, attention kernel, KV cache and prompt-cache flags. */
TernaryError args_parse_memory(CliArgs *args, int argc, char **argv, int *ip, int *handled);

/* remote layers, request queue, rate/timeout/connection limits and parallel slots. */
TernaryError args_parse_serving(CliArgs *args, int argc, char **argv, int *ip, int *handled);

/* best-of and tree-search (MCTS) flags and the reasoning judge. */
TernaryError args_parse_search(CliArgs *args, int argc, char **argv, int *ip, int *handled);

/* chat format, web UI branding, memory DB, calibration, numerics, output switches and the HTTP server. */
TernaryError args_parse_behavior(CliArgs *args, int argc, char **argv, int *ip, int *handled);

/* draft model, LoRA, colour and update-check flags. */
TernaryError args_parse_extras(CliArgs *args, int argc, char **argv, int *ip, int *handled);

#endif
