import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_log_secrets as c


class LogSecretTests(unittest.TestCase):
    def test_flags_a_secret_argument(self):
        self.assertEqual(
            c.violations('fprintf(stderr, "key=%s\\n", ctx->api_key);'), [(1, "api_key")]
        )

    def test_flags_a_multiline_call(self):
        text = 'fprintf(stderr,\n  "auth %s",\n  auth_header);'
        self.assertEqual(c.violations(text), [(3, "auth_header")])

    def test_help_text_in_a_string_is_fine(self):
        self.assertEqual(c.violations('printf("  --api-key <key>  Require Bearer <key>\\n");'), [])

    def test_comments_are_ignored(self):
        self.assertEqual(c.violations("/* printf(api_key); */\n// puts(password);"), [])

    def test_non_print_use_is_fine(self):
        self.assertEqual(c.violations("if (strcmp(tok, api_key) == 0) return 1;"), [])

    def test_perror_and_puts_are_covered(self):
        self.assertEqual(len(c.violations("puts(password); perror(passphrase);")), 2)

    def test_the_logger_calls_are_covered(self):
        self.assertEqual(
            c.violations('TN_LOG_ERR("auth", "key %s\\n", server_config.auth.api_key);'),
            [(1, "api_key")],
        )
        self.assertEqual(
            c.violations(
                'tn_log(TN_LOG_INFO, "x", "%s", auth_header); TN_OUT_INF("x", "%s", password);'
            ),
            [(1, "auth_header"), (1, "password")],
        )

    def test_the_real_tree_is_clean(self):
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()
