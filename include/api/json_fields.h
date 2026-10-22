#ifndef TN_JSON_FIELDS_H
#define TN_JSON_FIELDS_H

/*
 * json_fields.h — read individual top-level fields of a small JSON object
 * (for the simple endpoints: watermark detect, config). Shares the
 * string/number decoding of json_parse.c.
 */

/* Heap copy of a top-level string field, or NULL if absent / not a string. */
char *json_field_string(const char *json_body, const char *key);

/* 1 and *out set if the top-level field exists and is a number, else 0. */
int json_field_number(const char *json_body, const char *key, double *out);

/* 1 and *out (0/1) set if the top-level field exists and is a boolean, else 0. */
int json_field_bool(const char *json_body, const char *key, int *out);

/* Heap copy of a top-level number field's literal text ("0.7", "-1"), so a
 * value can be stored exactly as sent; NULL if absent / not a number. */
char *json_field_number_text(const char *json_body, const char *key);

/* Top-level array of strings: up to max heap copies into out[]. Returns the
 * count, or -1 if absent / not an array of strings (nothing allocated). */
int json_field_string_list(const char *json_body, const char *key, char **out, int max);

#endif /* TN_JSON_FIELDS_H */
