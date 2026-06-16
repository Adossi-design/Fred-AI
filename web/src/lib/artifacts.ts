// Detects a substantial code/markup block in an assistant message that is
// worth opening in the Canvas side panel (Artifacts-style).

export interface Artifact {
  title: string
  language: string
  code: string
  isMarkup: boolean
}

export function extractArtifact(content?: string): Artifact | null {
  if (!content) return null

  const re = /```([\w+-]*)\n([\s\S]*?)```/g
  let m: RegExpExecArray | null
  let best: { language: string; code: string; isMarkup: boolean } | null = null

  while ((m = re.exec(content)) !== null) {
    const lang = (m[1] || '').toLowerCase()
    const code = m[2].replace(/\s+$/, '')
    if (!code) continue

    const lines = code.split('\n').length
    const isMarkup =
      ['html', 'svg', 'xml'].includes(lang) ||
      /<!doctype html|<html[\s>]|<svg[\s>]/i.test(code)
    const substantial = isMarkup || lines >= 8 || code.length >= 280
    if (!substantial) continue

    // Prefer markup (it gets a live preview); otherwise keep the largest block.
    if (
      !best ||
      (isMarkup && !best.isMarkup) ||
      (isMarkup === best.isMarkup && code.length > best.code.length)
    ) {
      best = { language: lang || (isMarkup ? 'html' : 'text'), code, isMarkup }
    }
  }

  if (!best) return null
  const title = best.isMarkup
    ? 'Preview'
    : best.language && best.language !== 'text'
      ? best.language.toUpperCase()
      : 'Code'
  return { title, language: best.language, code: best.code, isMarkup: best.isMarkup }
}
