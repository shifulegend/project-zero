/** The message of whatever a `catch` caught (an Error, a plain object with a message, or any value). */
export function errorText(e: unknown): string {
  const message = (e as { message?: unknown } | null | undefined)?.message;
  return typeof message === 'string' ? message : String(e);
}
