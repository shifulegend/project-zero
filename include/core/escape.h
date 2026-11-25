#ifndef TN_ESCAPE_H
#define TN_ESCAPE_H

/* The one-character escapes JSON and TOML basic strings share: \" \\ \n \t \r \b \f.
 * Returns the byte the escape letter `e` stands for, or -1 when `e` is not one of them. The
 * parsers add their own extras (JSON: \/ and \uXXXX). */
int tn_escape_simple(char e);

#endif
