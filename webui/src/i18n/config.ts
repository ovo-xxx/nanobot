export const LOCALE_STORAGE_KEY = "nanobot.locale";

export const supportedLocales = [
  { code: "en", nativeLabel: "English" },
  { code: "zh-CN", nativeLabel: "简体中文" },
  { code: "zh-TW", nativeLabel: "繁體中文" },
  { code: "fr", nativeLabel: "Français" },
  { code: "ja", nativeLabel: "日本語" },
  { code: "ko", nativeLabel: "한국어" },
  { code: "es", nativeLabel: "Español" },
  { code: "pt-BR", nativeLabel: "Português (Brasil)" },
  { code: "vi", nativeLabel: "Tiếng Việt" },
  { code: "id", nativeLabel: "Bahasa Indonesia" },
] as const;

export type SupportedLocale = (typeof supportedLocales)[number]["code"];

export const defaultLocale: SupportedLocale = "en";
export const fallbackLocale: SupportedLocale = "en";

/**
 * Map a language tag onto a supported locale, or return null when the tag has no
 * supported locale behind it. Callers that must always land somewhere use
 * {@link normalizeLocale}; callers that need to tell "unsupported" apart from
 * "English" (the browser probe) use this.
 */
export function matchSupportedLocale(
  input: string | null | undefined,
): SupportedLocale | null {
  if (!input) return null;
  const trimmed = input.trim();
  if (!trimmed) return null;

  const exact = supportedLocales.find((locale) => locale.code === trimmed);
  if (exact) return exact.code;

  const lower = trimmed.toLowerCase();
  if (lower === "zh" || lower.startsWith("zh-cn") || lower.startsWith("zh-sg")) {
    return "zh-CN";
  }
  if (
    lower.startsWith("zh-tw") ||
    lower.startsWith("zh-hk") ||
    lower.startsWith("zh-mo") ||
    lower.startsWith("zh-hant")
  ) {
    return "zh-TW";
  }
  if (lower === "pt" || lower.startsWith("pt-")) {
    return "pt-BR";
  }

  const base = lower.split("-")[0];
  const baseMatch = supportedLocales.find(
    (locale) => locale.code.toLowerCase() === base,
  );
  return baseMatch?.code ?? null;
}

export function normalizeLocale(
  input: string | null | undefined,
): SupportedLocale {
  return matchSupportedLocale(input) ?? defaultLocale;
}

function readStoredLocale(): SupportedLocale | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    return raw ? normalizeLocale(raw) : null;
  } catch {
    return null;
  }
}

/**
 * The browser's own preference, as the preboot script in index.html reads it.
 *
 * That inline script localizes the splash screen from `navigator.languages[0] ||
 * navigator.language`, so the app has to consult the same value or a first visit
 * renders a Chinese splash and then flips the whole UI to English when React
 * mounts. Only the first non-empty candidate is used, matching the script exactly:
 * a browser whose top preference is unsupported stays on English in both places.
 */
function readBrowserLocale(): SupportedLocale | null {
  if (typeof navigator === "undefined") return null;
  try {
    const preferred = Array.isArray(navigator.languages)
      ? navigator.languages.find((tag) => typeof tag === "string" && tag.trim().length > 0)
      : undefined;
    return matchSupportedLocale(preferred ?? navigator.language);
  } catch {
    return null;
  }
}

export function resolveInitialLocale(): SupportedLocale {
  return readStoredLocale() ?? readBrowserLocale() ?? defaultLocale;
}

export function persistLocale(locale: SupportedLocale): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // ignore storage errors
  }
}

export function applyDocumentLocale(locale: SupportedLocale): void {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale;
}
