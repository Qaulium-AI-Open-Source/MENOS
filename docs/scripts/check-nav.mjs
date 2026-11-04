// Fails the documentation build on the structural mistakes a link checker
// cannot see: a sidebar entry whose page does not exist, a page that no
// sidebar entry reaches, an empty page, or placeholder text.
//
// Dead internal links inside pages are caught separately by VitePress itself
// (ignoreDeadLinks is left at its default, false).
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const EXCLUDED_TOP = new Set(['adr', 'benchmarks', 'node_modules', '.vitepress', 'public', 'scripts'])
const EXCLUDED_FILES = new Set(['BUILD_PLAN.md'])

function pages(dir = root, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    const rel = relative(root, full)
    if (statSync(full).isDirectory()) {
      if (dir === root && EXCLUDED_TOP.has(name)) continue
      pages(full, out)
    } else if (name.endsWith('.md') && !EXCLUDED_FILES.has(rel)) {
      out.push(rel.split(sep).join('/'))
    }
  }
  return out
}

const configSource = readFileSync(join(root, '.vitepress', 'config.mts'), 'utf8')
const linked = new Set(
  [...configSource.matchAll(/link:\s*'(\/[^']*)'/g)].map((m) => m[1].replace(/\/$/, '')),
)

const problems = []
const all = pages()

for (const file of all) {
  const route = '/' + file.replace(/\.md$/, '').replace(/(^|\/)index$/, '')
  const normalised = route === '/' ? '' : route
  if (!linked.has(normalised) && file !== 'index.md') {
    problems.push(`orphan page (not reachable from the nav or sidebar): ${file}`)
  }
  const text = readFileSync(join(root, file), 'utf8')
  const body = text.replace(/^---[\s\S]*?---/, '').trim()
  if (body.length < 200) problems.push(`page is effectively empty: ${file}`)
  for (const banned of [/\bTODO\b/, /\bTBD\b/i, /coming soon/i, /lorem ipsum/i, /\bFIXME\b/]) {
    if (banned.test(text)) problems.push(`placeholder text (${banned}) in ${file}`)
  }
}

for (const target of linked) {
  if (target === '' || target === '/') continue
  const candidates = [`${target.slice(1)}.md`, `${target.slice(1)}/index.md`]
  if (!candidates.some((c) => existsSync(join(root, c)))) {
    problems.push(`navigation entry has no page: ${target}`)
  }
}

if (problems.length) {
  console.error('Documentation structure check FAILED:\n  - ' + problems.join('\n  - '))
  process.exit(1)
}
console.log(`Documentation structure check passed: ${all.length} pages, all reachable, none empty.`)
