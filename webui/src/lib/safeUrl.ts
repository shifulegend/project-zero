// Links that come from the server (release notes, changelog) are only shown as links when they
// are plain https:// URLs. A javascript:, data: or http: value is dropped instead of being put in
// an href.

/** The URL when it is a well-formed https URL, else null. */
export function safeHttpsUrl(url: unknown): string | null {
  if (typeof url !== 'string') return null;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  return parsed.protocol === 'https:' && parsed.hostname !== '' ? parsed.href : null;
}
