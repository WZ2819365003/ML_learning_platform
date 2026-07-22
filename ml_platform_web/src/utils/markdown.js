/**
 * Minimal Markdown parser for the report subset (M3-1).
 *
 * Deliberately not react-markdown. Two reasons:
 *
 * 1. **We own the input.** `report_service.build_task_report` emits a fixed,
 *    small grammar — headings, GFM pipe tables, blockquotes, paragraphs, and
 *    inline `**bold**` / `` `code` ``. A general CommonMark engine solves
 *    problems this input never poses.
 * 2. **Lockfile risk.** Adding dependencies here has broken this repo before:
 *    running `npm install` on macOS pruned the linux-only platform packages
 *    from package-lock.json and the Docker build stopped resolving modules.
 *
 * This parser produces *tokens*, never HTML. The renderer turns tokens into
 * React elements, so React escapes the content for us. That matters: the
 * report interpolates user-controlled strings (dataset names, feature names,
 * model names). A string-concatenating renderer with dangerouslySetInnerHTML
 * would turn a dataset named `<img onerror=...>` into stored XSS.
 */

/** Split a GFM table row into trimmed cells, dropping the outer pipes. */
function splitRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

const ALIGNMENT_ROW = /^\|?[\s:|-]+\|[\s:|-]*$/

function isTableRow(line) {
  return line.trim().startsWith('|')
}

/**
 * Parse Markdown into a flat list of block tokens.
 *
 * Returns blocks of shape:
 *   { type: 'heading',    level: 1..3, text }
 *   { type: 'table',      headers: string[], rows: string[][] }
 *   { type: 'blockquote', text }
 *   { type: 'paragraph',  text }
 */
export function parseMarkdown(markdown) {
  const lines = String(markdown ?? '').split('\n')
  const blocks = []
  let i = 0

  const flushParagraph = (buffer) => {
    const text = buffer.join(' ').trim()
    if (text) blocks.push({ type: 'paragraph', text })
  }

  let paragraph = []

  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!trimmed) {
      flushParagraph(paragraph)
      paragraph = []
      i += 1
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flushParagraph(paragraph)
      paragraph = []
      blocks.push({
        type: 'heading',
        level: Math.min(heading[1].length, 6),
        text: heading[2].trim(),
      })
      i += 1
      continue
    }

    if (trimmed.startsWith('>')) {
      flushParagraph(paragraph)
      paragraph = []
      const quote = []
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quote.push(lines[i].trim().replace(/^>\s?/, ''))
        i += 1
      }
      blocks.push({ type: 'blockquote', text: quote.join(' ').trim() })
      continue
    }

    // A table needs a header row followed by an alignment row; anything else
    // starting with "|" is just text and must not silently become a table.
    if (isTableRow(line) && i + 1 < lines.length && ALIGNMENT_ROW.test(lines[i + 1])) {
      flushParagraph(paragraph)
      paragraph = []
      const headers = splitRow(line)
      i += 2
      const rows = []
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(splitRow(lines[i]))
        i += 1
      }
      blocks.push({ type: 'table', headers, rows })
      continue
    }

    paragraph.push(trimmed)
    i += 1
  }

  flushParagraph(paragraph)
  return blocks
}

/**
 * Split inline text into segments for the renderer.
 *
 * Returns `{ kind: 'text' | 'bold' | 'code', value }[]`. Values stay raw —
 * escaping is the renderer's job, and React does it by construction.
 */
export function parseInline(text) {
  const segments = []
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let cursor = 0
  let match

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      segments.push({ kind: 'text', value: text.slice(cursor, match.index) })
    }
    const token = match[0]
    if (token.startsWith('**')) {
      // Bold routinely wraps a code span — the report's headline is
      // `**结论：选定模型为 \`xgboost\`…**`. Without recursing, the backticks
      // render literally inside the bold text. Recursion terminates because
      // the bold body cannot itself contain `**`.
      const value = token.slice(2, -2)
      segments.push({ kind: 'bold', value, children: parseInline(value) })
    } else {
      segments.push({ kind: 'code', value: token.slice(1, -1) })
    }
    cursor = match.index + token.length
  }

  if (cursor < text.length) {
    segments.push({ kind: 'text', value: text.slice(cursor) })
  }
  return segments
}
