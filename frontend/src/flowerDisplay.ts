export function formatFlowerDisplayName(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return value

  return trimmed
    .toLowerCase()
    .replace(/(^|[\s\-/(])([a-z])/g, (_match, prefix: string, letter: string) => `${prefix}${letter.toUpperCase()}`)
}
