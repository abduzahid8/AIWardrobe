export function parseIncomingProduct(input: string): {
  url?: string;
  title?: string;
  error?: string;
} {
  if (/^(javascript|data|file|ftp|intent):/i.test(input.trim()))
    return { error: 'Unsupported product URL scheme' };
  const trimmed = input.trim().slice(0, 4096),
    raw = trimmed.match(/https?:\/\/[^\s<>"']+/i)?.[0]?.replace(/[),.;]+$/, '');
  if (!raw)
    return /\w|[\p{L}]/u.test(trimmed) && !/^[a-z]+:\/\//i.test(trimmed)
      ? { title: trimmed.slice(0, 200) }
      : { error: 'Enter a product URL or title' };
  try {
    const url = new URL(raw),
      host = url.hostname.toLowerCase();
    if (
      !['https:', 'http:'].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.port ||
      host === 'localhost' ||
      !host.includes('.') ||
      /\.(local|internal|localhost)$/.test(host) ||
      /^\d+\.\d+\.\d+\.\d+$/.test(host) ||
      host.includes(':') ||
      host.includes('[')
    )
      return { error: 'Only public merchant URLs are supported' };
    url.hash = '';
    for (const key of [...url.searchParams.keys()])
      if (
        /^utm_|^fbclid$|^gclid$|^(access_token|refresh_token|session|sessionid|password|csrf|_csrf)$/i.test(
          key
        )
      )
        url.searchParams.delete(key);
    return { url: url.toString() };
  } catch {
    return { error: 'Invalid product URL' };
  }
}
