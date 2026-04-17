import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from 'react'
import { animate, createScope, stagger } from 'animejs'
import type { Target } from 'animejs'
import './App.css'
import SearchIcon from './assets/mag.png'
import FlowerCoral from './assets/flower-coral.svg'
import FlowerGold from './assets/flower-gold.svg'
import FlowerOlive from './assets/flower-olive.svg'
import FlowerRose from './assets/flower-rose.svg'
import { formatFlowerDisplayName } from './flowerDisplay'
import { AutocompleteResponse, KeywordUsed, RecommendationResponse } from './types'

const EMPTY_RESULTS: RecommendationResponse = {
  query: '',
  keywords_used: [],
  score_scale: 'unit',
  query_latent_radar_chart: null,
  query_latent_radar_axes: [],
  suggestions: [],
}

const ANIMATED_QUERY_EXAMPLES = [
  'white flowers for gratitude',
  'low maintenance pink flowers',
  'yellow flowers that mean friendship',
]

type FlowerTone = 'coral' | 'gold' | 'olive' | 'rose'

interface DecorativeFlower {
  top: string
  left?: string
  right?: string
  size: number
  depth: number
  hue: FlowerTone
  image: string
}

const DECORATIVE_FLOWERS: DecorativeFlower[] = [
  { top: '6%', left: '4%', size: 118, depth: 1.25, hue: 'coral', image: FlowerCoral },
  { top: '15%', right: '9%', size: 92, depth: 0.92, hue: 'gold', image: FlowerGold },
  { top: '28%', left: '1.5%', size: 84, depth: 1.08, hue: 'olive', image: FlowerOlive },
  { top: '40%', right: '3%', size: 108, depth: 1.36, hue: 'rose', image: FlowerRose },
  { top: '53%', left: '7%', size: 98, depth: 1.12, hue: 'coral', image: FlowerCoral },
  { top: '63%', right: '12%', size: 90, depth: 1.03, hue: 'gold', image: FlowerGold },
  { top: '74%', left: '10%', size: 102, depth: 1.22, hue: 'olive', image: FlowerOlive },
  { top: '82%', right: '5%', size: 96, depth: 1.14, hue: 'rose', image: FlowerRose },
  { top: '10%', left: '18%', size: 74, depth: 0.88, hue: 'gold', image: FlowerGold },
  { top: '21%', right: '22%', size: 72, depth: 0.84, hue: 'coral', image: FlowerCoral },
  { top: '33%', left: '14%', size: 78, depth: 0.9, hue: 'rose', image: FlowerRose },
  { top: '47%', right: '18%', size: 70, depth: 0.86, hue: 'olive', image: FlowerOlive },
  { top: '58%', left: '18%', size: 76, depth: 0.89, hue: 'gold', image: FlowerGold },
  { top: '69%', right: '24%', size: 74, depth: 0.9, hue: 'coral', image: FlowerCoral },
  { top: '79%', left: '24%', size: 68, depth: 0.82, hue: 'rose', image: FlowerRose },
  { top: '87%', right: '20%', size: 72, depth: 0.85, hue: 'olive', image: FlowerOlive },
]

interface AppProps {
  isActive?: boolean
}

function formatLabel(values: string[] | undefined): string {
  if (!values || values.length === 0) return 'Not listed'
  return values.join(', ')
}

function formatMaintenanceLabel(values: string[] | undefined): string {
  if (!values || values.length === 0) return 'Not listed'

  return values
    .map((value) => {
      const trimmed = value.trim()
      const normalized = trimmed.toLowerCase()
      if (normalized === 'low' || normalized === 'medium' || normalized === 'high') {
        return `${normalized} maintenance`
      }
      return trimmed
    })
    .join(', ')
}

function formatFullText(values: string[]): string {
  if (values.length === 0) return 'Not listed'
  const combined = values.join(' ').replace(/\s+/g, ' ').trim()
  return combined || 'Not listed'
}

function detailExpandKey(
  suggestionKey: string,
  section: 'meaning' | 'occasion' | 'details',
): string {
  return `${suggestionKey}::${section}`
}

interface HighlightProfile {
  normalized: string
  root: string
}

function normalizeHighlightWord(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}

function stemHighlightWord(value: string): string {
  let normalized = normalizeHighlightWord(value)
  if (normalized.length <= 4) return normalized

  const suffixRules: Array<[string, string]> = [
    ['fulness', 'ful'],
    ['ousness', 'ous'],
    ['ization', 'ize'],
    ['ations', 'ate'],
    ['ation', 'ate'],
    ['lessly', 'less'],
    ['ship', ''],
    ['ment', ''],
    ['ingly', ''],
    ['fully', 'ful'],
    ['iness', 'y'],
    ['edly', ''],
    ['able', ''],
    ['ible', ''],
    ['ness', ''],
    ['less', ''],
    ['iest', 'y'],
    ['ier', 'y'],
    ['ies', 'y'],
    ['ing', ''],
    ['er', ''],
    ['est', ''],
    ['ed', ''],
    ['ly', ''],
    ['ful', ''],
    ['ous', ''],
    ['ive', ''],
    ['al', ''],
    ['ity', ''],
    ['s', ''],
  ]

  for (const [suffix, replacement] of suffixRules) {
    if (!normalized.endsWith(suffix)) continue

    const nextValue = normalized.slice(0, -suffix.length) + replacement
    if (nextValue.length >= 4) {
      normalized = nextValue
      break
    }
  }

  return normalized
}

function getCommonPrefixLength(left: string, right: string): number {
  const limit = Math.min(left.length, right.length)
  let index = 0

  while (index < limit && left[index] === right[index]) {
    index += 1
  }

  return index
}

function areSimilarHighlightWords(left: HighlightProfile, right: HighlightProfile): boolean {
  if (left.normalized === right.normalized || left.root === right.root) {
    return true
  }

  const prefixLength = getCommonPrefixLength(left.root, right.root)
  const shorterLength = Math.min(left.root.length, right.root.length)
  return shorterLength >= 4 && prefixLength >= Math.max(4, Math.ceil(shorterLength * 0.6))
}

function getHighlightTerms(values: string[]): HighlightProfile[] {
  const deduped = Array.from(
    new Set(
      values
        .flatMap((value) => value.split(/[^a-z0-9]+/i))
        .map((value) => normalizeHighlightWord(value))
        .filter((value) => value.length >= 2),
    ),
  )

  return deduped
    .sort((left, right) => right.length - left.length)
    .map((value) => ({
      normalized: value,
      root: stemHighlightWord(value),
    }))
}

function renderHighlightedText(text: string, highlightTerms: HighlightProfile[]): ReactNode {
  if (!text || text === 'Not listed' || highlightTerms.length === 0) {
    return text
  }

  const pattern = /([a-z0-9]+)/gi
  const parts = text.split(pattern)

  return parts.map((part, index) => {
    const normalizedPart = normalizeHighlightWord(part)
    const isMatch =
      normalizedPart.length >= 2 &&
      highlightTerms.some((term) =>
        areSimilarHighlightWords(term, {
          normalized: normalizedPart,
          root: stemHighlightWord(normalizedPart),
        }),
      )

    return isMatch ? (
      <mark key={`${part}-${index}`} className="detail-highlight">
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    )
  })
}

// Relative to top result
/*
function getMatchStrength(score: number, bestScore: number): number {
  if (bestScore <= 0) return 0
  return Math.max(Math.min(score / bestScore, 1), 0.12)
}
  */

function getMatchStrength(score: number, scoreScale: RecommendationResponse['score_scale']): number {
  if (scoreScale === 'unit') {
    return Math.max(Math.min(score, 1), 0)
  }
  return Math.max(Math.min(score / 100, 1), 0)
}

function formatMatchStrengthLabel(
  strength: number,
  scoreScale: RecommendationResponse['score_scale'],
): string {
  if (scoreScale === 'unit') {
    return `${Math.round(strength * 100)}%`
  }
  return `${Math.round(strength * 100)}%`
}

function getQueryBreakdownKeywords(results: RecommendationResponse): KeywordUsed[] {
  if (results.keywords_used.length > 0) {
    return results.keywords_used
  }

  const breakdownKeywords = new Map<string, KeywordUsed & { firstSeenAt: number }>()
  let firstSeenAt = 0

  for (const suggestion of results.suggestions) {
    for (const match of suggestion.matched_keywords) {
      const keyword = match.keyword.trim()
      if (!keyword) continue

      const key = `${match.category}:${keyword.toLowerCase()}`
      const current = breakdownKeywords.get(key)
      if (!current) {
        breakdownKeywords.set(key, {
          keyword,
          category: match.category,
          score: match.score,
          firstSeenAt,
        })
        firstSeenAt += 1
        continue
      }

      current.score = Math.max(current.score, match.score)
    }
  }

  const normalizedKeywords = Array.from(breakdownKeywords.values())
    .sort((left, right) => right.score - left.score || left.firstSeenAt - right.firstSeenAt)
    .map(({ firstSeenAt: _firstSeenAt, ...keyword }) => keyword)

  if (normalizedKeywords.length === 0) {
    return []
  }
  return normalizedKeywords.slice(0, 10)
}

function App({ isActive = true }: AppProps): JSX.Element {
  const [query, setQuery] = useState<string>('')
  const [isQueryFocused, setIsQueryFocused] = useState<boolean>(false)
  const [animatedQueryText, setAnimatedQueryText] = useState<string>('')
  const [animatedQueryIndex, setAnimatedQueryIndex] = useState<number>(0)
  const [animatedQueryDeleting, setAnimatedQueryDeleting] = useState<boolean>(false)
  const [results, setResults] = useState<RecommendationResponse>(EMPTY_RESULTS)
  const [autocompleteSuggestions, setAutocompleteSuggestions] = useState<string[]>([])
  const [autocompleteOpen, setAutocompleteOpen] = useState<boolean>(false)
  const [autocompleteLoading, setAutocompleteLoading] = useState<boolean>(false)
  const [autocompleteEnabled, setAutocompleteEnabled] = useState<boolean>(false)
  const [activeAutocompleteIndex, setActiveAutocompleteIndex] = useState<number>(-1)
  const [expandedDetailSections, setExpandedDetailSections] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [resultLimit, setResultLimit] = useState<number>(5)
  const [appliedLimit, setAppliedLimit] = useState<number>(5)
  const [searchMethod, setSearchMethod] = useState<'svd' | 'tfidf'>('svd')
  const queryInputRef = useRef<HTMLInputElement | null>(null)
  const appRef = useRef<HTMLElement | null>(null)
  const animationScopeRef = useRef<ReturnType<typeof createScope> | null>(null)
  const loadingAnimationRef = useRef<ReturnType<typeof animate> | null>(null)
  const autocompleteRequestRef = useRef<number>(0)
  const skipNextAutocompleteRef = useRef<boolean>(false)

  const queryBreakdownKeywords = getQueryBreakdownKeywords(results)
  // const topScore = results.suggestions[0]?.score ?? 0
  const showAnimatedQuery = isActive && !query && !isQueryFocused

  useEffect(() => {
    if (!showAnimatedQuery) {
      if (animatedQueryText) {
        setAnimatedQueryText('')
      }
      if (animatedQueryDeleting) {
        setAnimatedQueryDeleting(false)
      }
      return
    }

    const currentExample = ANIMATED_QUERY_EXAMPLES[animatedQueryIndex % ANIMATED_QUERY_EXAMPLES.length]
    const hasTypedFullExample = animatedQueryText === currentExample
    const hasDeletedEverything = animatedQueryText.length === 0

    const timeoutId = window.setTimeout(() => {
      if (!animatedQueryDeleting && !hasTypedFullExample) {
        setAnimatedQueryText(currentExample.slice(0, animatedQueryText.length + 1))
        return
      }

      if (!animatedQueryDeleting && hasTypedFullExample) {
        setAnimatedQueryDeleting(true)
        return
      }

      if (animatedQueryDeleting && !hasDeletedEverything) {
        setAnimatedQueryText(currentExample.slice(0, Math.max(animatedQueryText.length - 1, 0)))
        return
      }

      setAnimatedQueryDeleting(false)
      setAnimatedQueryIndex((currentIndex) => (currentIndex + 1) % ANIMATED_QUERY_EXAMPLES.length)
    }, !animatedQueryDeleting && !hasTypedFullExample ? 68 : !animatedQueryDeleting ? 1250 : hasDeletedEverything ? 240 : 34)

    return () => window.clearTimeout(timeoutId)
  }, [animatedQueryDeleting, animatedQueryIndex, animatedQueryText, showAnimatedQuery])

  useEffect(() => {
    if (!isActive) {
      loadingAnimationRef.current?.revert()
      animationScopeRef.current?.revert()
      animationScopeRef.current = null
      return
    }

    if (!appRef.current) return

    const scope = createScope({
      root: appRef,
      mediaQueries: {
        compact: '(max-width: 760px)',
        reduceMotion: '(prefers-reduced-motion: reduce)',
      },
    })

    scope.add('animateKeywords', () => {
      if (scope.matches.reduceMotion) return

      animate('.keyword-card', {
        opacity: [0, 1],
        translateY: [18, 0],
        scale: [0.9, 1],
        delay: stagger(scope.matches.compact ? 55 : 85),
        duration: scope.matches.compact ? 380 : 540,
        ease: 'out(4)',
      })
    })

    scope.add('animateSuggestions', () => {
      if (scope.matches.reduceMotion) return

      animate('.suggestion-card', {
        opacity: [0, 1],
        translateY: [scope.matches.compact ? 26 : 42, 0],
        scale: [0.94, 1],
        delay: stagger(scope.matches.compact ? 90 : 130),
        duration: scope.matches.compact ? 520 : 760,
        ease: 'out(4)',
      })

      animate('.score-fill', {
        scaleX: (target: Target) => [0, Number((target as HTMLElement).dataset.strength ?? '0')],
        delay: stagger(scope.matches.compact ? 90 : 130, { start: scope.matches.compact ? 120 : 180 }),
        duration: scope.matches.compact ? 680 : 900,
        ease: 'out(5)',
      })

      animate('.match-chip', {
        opacity: [0, 1],
        translateY: [10, 0],
        delay: stagger(28, { start: scope.matches.compact ? 180 : 260 }),
        duration: 320,
        ease: 'out(3)',
      })
    })

    scope.add('startLoading', () => {
      if (scope.matches.reduceMotion) return

      loadingAnimationRef.current?.revert()
      loadingAnimationRef.current = animate('.loading-dot', {
        translateY: ['0rem', '-0.75rem'],
        scale: [0.82, 1.08],
        opacity: [0.35, 1],
        delay: stagger(scope.matches.compact ? 100 : 140),
        duration: scope.matches.compact ? 540 : 760,
        ease: 'inOutSine',
        loop: true,
        alternate: true,
      })
    })

    scope.add('stopLoading', () => {
      loadingAnimationRef.current?.revert()
      loadingAnimationRef.current = null
    })

    scope.add(() => {
      if (scope.matches.reduceMotion || scope.matches.compact || !appRef.current) return

      const root = appRef.current
      const flowers = Array.from(root.querySelectorAll<HTMLElement>('.bg-flower'))
      if (flowers.length === 0) return

      let pointerX = window.innerWidth * 0.5
      let pointerY = window.innerHeight * 0.5
      let frameId = 0
      const offsets = flowers.map(() => ({ x: 0, y: 0 }))

      animate('.bg-flower', {
        opacity: [0, 1],
        scale: [0.86, 1],
        delay: stagger(120),
        duration: 900,
        ease: 'out(4)',
      })

      const render = (): void => {
        flowers.forEach((flower, index) => {
          const rect = flower.getBoundingClientRect()
          const centerX = rect.left + rect.width / 2
          const centerY = rect.top + rect.height / 2
          const dx = centerX - pointerX
          const dy = centerY - pointerY
          const distance = Math.hypot(dx, dy) || 1
          const radius = rect.width * 0.95 + 90
          const strength = Math.max(0, 1 - distance / radius)
          const depth = Number(flower.dataset.depth ?? '1')
          const desiredX = strength > 0 ? (dx / distance) * strength * 42 * depth : 0
          const desiredY = strength > 0 ? (dy / distance) * strength * 34 * depth : 0

          offsets[index].x += (desiredX - offsets[index].x) * 0.16
          offsets[index].y += (desiredY - offsets[index].y) * 0.16

          flower.style.transform = `translate3d(${offsets[index].x}px, ${offsets[index].y}px, 0)`
        })

        frameId = window.requestAnimationFrame(render)
      }

      const handlePointerMove = (event: PointerEvent): void => {
        pointerX = event.clientX
        pointerY = event.clientY
      }

      const handlePointerLeave = (): void => {
        pointerX = -1000
        pointerY = -1000
      }

      frameId = window.requestAnimationFrame(render)
      window.addEventListener('pointermove', handlePointerMove)
      window.addEventListener('pointerleave', handlePointerLeave)

      return () => {
        window.cancelAnimationFrame(frameId)
        window.removeEventListener('pointermove', handlePointerMove)
        window.removeEventListener('pointerleave', handlePointerLeave)
      }
    })

    animationScopeRef.current = scope

    return () => {
      loadingAnimationRef.current?.revert()
      animationScopeRef.current?.revert()
      animationScopeRef.current = null
    }
  }, [isActive])

  useEffect(() => {
    const scope = animationScopeRef.current
    if (!scope) return

    if (loading) {
      scope.methods.startLoading?.()
      return
    }

    scope.methods.stopLoading?.()
  }, [loading])

  useEffect(() => {
    const scope = animationScopeRef.current
    if (!scope || loading) return

    if (queryBreakdownKeywords.length > 0) {
      scope.methods.animateKeywords?.()
    }

    if (results.suggestions.length > 0) {
      scope.methods.animateSuggestions?.()
    }
  }, [queryBreakdownKeywords.length, results.suggestions.length, loading])

  useEffect(() => {
    if (!autocompleteEnabled) {
      setAutocompleteOpen(false)
      setAutocompleteLoading(false)
      return
    }

    if (skipNextAutocompleteRef.current) {
      skipNextAutocompleteRef.current = false
      return
    }

    const trimmedQuery = query.trim()
    if (trimmedQuery.length < 2) {
      setAutocompleteSuggestions([])
      setAutocompleteOpen(false)
      setAutocompleteLoading(false)
      setActiveAutocompleteIndex(-1)
      return
    }

    const requestId = autocompleteRequestRef.current + 1
    autocompleteRequestRef.current = requestId
    const controller = new AbortController()
    const timeoutId = window.setTimeout(async () => {
      setAutocompleteLoading(true)
      try {
        const response = await fetch(`/api/autocomplete?q=${encodeURIComponent(trimmedQuery)}`, {
          signal: controller.signal,
        })
        if (!response.ok) {
          throw new Error(`Autocomplete failed with status ${response.status}`)
        }

        const data: AutocompleteResponse = await response.json()
        if (autocompleteRequestRef.current !== requestId) return

        setAutocompleteSuggestions(data.suggestions)
        setAutocompleteOpen(data.suggestions.length > 0)
        setActiveAutocompleteIndex(-1)
      } catch (requestError) {
        if ((requestError as Error).name === 'AbortError') return
        if (autocompleteRequestRef.current !== requestId) return
        setAutocompleteSuggestions([])
        setAutocompleteOpen(false)
      } finally {
        if (autocompleteRequestRef.current === requestId) {
          setAutocompleteLoading(false)
        }
      }
    }, 180)

    return () => {
      controller.abort()
      window.clearTimeout(timeoutId)
    }
  }, [query])

  const runSearch = async (nextQuery: string): Promise<void> => {
    const trimmedQuery = nextQuery.trim()
    skipNextAutocompleteRef.current = true
    setAutocompleteEnabled(false)
    setQuery(nextQuery)
    setAutocompleteSuggestions([])
    setAutocompleteOpen(false)
    setAutocompleteLoading(false)
    setActiveAutocompleteIndex(-1)
    queryInputRef.current?.blur()
    setExpandedDetailSections({})

    if (!trimmedQuery) {
      setResults(EMPTY_RESULTS)
      setError('')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(`/api/recommendations?q=${encodeURIComponent(trimmedQuery)}&limit=${resultLimit}&method=${searchMethod}`)
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data: RecommendationResponse = await response.json()
      setResults(data)
      setAppliedLimit(resultLimit)
    } catch (requestError) {
      setResults(EMPTY_RESULTS)
      setError(requestError instanceof Error ? requestError.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const applyAutocompleteSuggestion = async (suggestion: string): Promise<void> => {
    await runSearch(suggestion)
  }

  const handleQueryChange = (value: string): void => {
    setQuery(value)
    if (!value.trim()) {
      setAutocompleteSuggestions([])
      setAutocompleteOpen(false)
      setActiveAutocompleteIndex(-1)
    }
  }

  const handleQueryKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (!autocompleteOpen || autocompleteSuggestions.length === 0) return

    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveAutocompleteIndex((currentIndex) => Math.min(currentIndex + 1, autocompleteSuggestions.length - 1))
      return
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveAutocompleteIndex((currentIndex) => Math.max(currentIndex - 1, 0))
      return
    }

    if (event.key === 'Escape') {
      setAutocompleteOpen(false)
      setActiveAutocompleteIndex(-1)
      return
    }

    if (event.key === 'Enter' && activeAutocompleteIndex >= 0) {
      event.preventDefault()
      void applyAutocompleteSuggestion(autocompleteSuggestions[activeAutocompleteIndex])
    }
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    await runSearch(query)
  }

  return (
    <main ref={appRef} className="app-shell">
      <div className="garden-backdrop" aria-hidden="true">
        {DECORATIVE_FLOWERS.map((flower, index) => (
          <div
            key={`${flower.top}-${flower.left ?? flower.right}-${index}`}
            className={`bg-flower ${flower.hue}`}
            data-depth={flower.depth}
            style={{
              top: flower.top,
              left: flower.left,
              right: flower.right,
              width: `${flower.size}px`,
              height: `${flower.size}px`,
            }}
          >
            <img className="bg-flower-image" src={flower.image} alt="" />
          </div>
        ))}
      </div>

      <div className="ranker-shell">
        <section className="ranker-intro">
          <div className="hero-copy-block">
            <p className="eyebrow">CREATE YOUR QUERY:</p>
            <p className="hero-copy ranker-copy">
              Describe a mood, color, level of care, and/or meaning, and receive a ranked set of flowers matched to your descriptions.
            </p>

            <form className="search-panel" onSubmit={handleSubmit}>
              <div className="search-row">
                <img src={SearchIcon} alt="" aria-hidden="true" />
                <div className="search-input-shell">
                  {showAnimatedQuery && (
                    <span className="search-ghost-query" aria-hidden="true">
                      <span className="search-ghost-prefix">e.g. </span>
                      {animatedQueryText}
                    </span>
                  )}
                  <input
                    ref={queryInputRef}
                    id="flower-query"
                    value={query}
                    onChange={(event) => handleQueryChange(event.target.value)}
                    onKeyDown={handleQueryKeyDown}
                    onFocus={() => {
                      setIsQueryFocused(true)
                      setAutocompleteEnabled(true)
                      if (autocompleteSuggestions.length > 0) {
                        setAutocompleteOpen(true)
                      }
                    }}
                    onBlur={() => {
                      setIsQueryFocused(false)
                      window.setTimeout(() => setAutocompleteOpen(false), 120)
                    }}
                    placeholder=""
                    autoComplete="off"
                    aria-autocomplete="list"
                    aria-expanded={autocompleteOpen}
                    aria-controls="flower-autocomplete-list"
                  />
                </div>
                <button type="submit" disabled={loading}>
                  {loading ? 'Ranking...' : 'Search'}
                </button>
              </div>
              <div className="search-options-row">
                <div className="limit-selector">
                  <span className="limit-label">Flowers Per Search:</span>
                  {[3, 5, 10].map(n => (
                    <button
                      key={n}
                      type="button"
                      className={`limit-chip ${resultLimit === n ? 'is-active' : ''}`}
                      onClick={() => setResultLimit(n)}
                    >
                      {n}
                    </button>
                  ))}
                </div>
                <div className="method-toggle">
                  <span className="limit-label">Retrieval Method:</span>
                  <button
                    type="button"
                    className={`method-chip ${searchMethod === 'svd' ? 'is-active' : ''}`}
                    onClick={() => setSearchMethod('svd')}
                  >
                    SVD
                  </button>
                  <button
                    type="button"
                    className={`method-chip ${searchMethod === 'tfidf' ? 'is-active' : ''}`}
                    onClick={() => setSearchMethod('tfidf')}
                  >
                    TF-IDF
                  </button>
                </div>
              </div>
            </form>

            <div
              className={`sample-row ${(autocompleteOpen || autocompleteLoading) ? 'is-autocomplete' : ''}`}
              id="flower-autocomplete-list"
              role={(autocompleteOpen || autocompleteLoading) ? 'listbox' : undefined}
            >
              <div className={`autocomplete-content ${(autocompleteOpen || autocompleteLoading) ? 'is-open' : ''}`}>
                <>
                  {autocompleteSuggestions.map((suggestion, index) => (
                    <button
                      key={suggestion}
                      type="button"
                      role="option"
                      aria-selected={index === activeAutocompleteIndex}
                      className={`autocomplete-option ${index === activeAutocompleteIndex ? 'is-active' : ''}`}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => void applyAutocompleteSuggestion(suggestion)}
                    >
                      <span>{suggestion}</span>
                    </button>
                  ))}
                  {autocompleteLoading && autocompleteSuggestions.length === 0 && (
                    <div className="autocomplete-empty">Loading suggestions...</div>
                  )}
                </>
              </div>
            </div>
          </div>

          <aside className="hero-side">
            <div className="hero-note">
              <span>How it works:</span>
              <p>
                The recommender projects flowers and queries into your chosen retrieval method space built
                from meanings, attributes, and flower descriptions, then ranks the closest matches.
              </p>
            </div>
          </aside>
        </section>

        <section className="page-body">
        <aside className="query-rail">
          <div className="rail-heading">
            <p>Query Breakdown</p>
            <span>{queryBreakdownKeywords.length}</span>
          </div>

          {!loading && results.query_latent_radar_chart && (
            <figure className="query-radar-panel">
              <div className="query-radar-head">
                <span>Latent Dimensions</span>
              </div>
              <img
                src={results.query_latent_radar_chart}
                alt={`query latent radar chart. axes: ${results.query_latent_radar_axes.join(', ')}`}
              />
            </figure>
          )}

          {loading ? (
            <div className="rail-empty">
              Projecting the query into the flower space.
            </div>
          ) : queryBreakdownKeywords.length > 0 ? (
            <div className="keyword-chip-list">
              {queryBreakdownKeywords.map((keyword) => (
                <div
                key={`${keyword.category}-${keyword.keyword}`}
                className="match-chip match-chip--rail"
              >
                <span>{keyword.keyword}</span>
                <small>{keyword.category}</small>
              </div>
              ))}
            </div>
          ) : (
            <div className="rail-empty">
              {error || 'Run a search to see which terms influenced the ranking!'}
            </div>
          )}
        </aside>

        <div className="results-column">
          <header className="results-header">
            <div className="results-title-block">
              <h2>Best Matches for Your Query: </h2>
              {results.query && !loading && <p>“{results.query}”</p>}
            </div>
            <div className="results-meta">
            <span>{results.suggestions.length}/{appliedLimit} shown</span>            
            </div>
          </header>

          {loading ? (
            <div className="loading-panel" aria-live="polite">
              <div className="loading-bloom" aria-hidden="true">
                <span className="loading-dot" />
                <span className="loading-dot" />
                <span className="loading-dot" />
                <span className="loading-dot" />
              </div>
              <strong>Ranking flowers</strong>
              <p>Encoding the query, searching the latent flower space, and sorting the shortlist.</p>
            </div>
          ) : results.suggestions.length > 0 ? (
            <div className="results-stream">
              {results.suggestions.map((suggestion, index) => {
                const strength = getMatchStrength(suggestion.score, results.score_scale ?? 'percent')
                const strengthLabel = formatMatchStrengthLabel(strength, results.score_scale ?? 'percent')
                const suggestionKey = `${suggestion.name}-${suggestion.scientific_name}`
                const displayName = formatFlowerDisplayName(suggestion.name)
                const fullMeaningText = formatFullText(suggestion.meanings)
                const fullOccasionText = formatFullText(suggestion.occasions ?? [])
                const meaningIsExpandable = fullMeaningText !== 'Not listed' && fullMeaningText.length > 220
                const occasionIsExpandable = fullOccasionText !== 'Not listed' && fullOccasionText.length > 220
                const isMeaningExpanded = Boolean(expandedDetailSections[detailExpandKey(suggestionKey, 'meaning')])
                const isOccasionExpanded = Boolean(expandedDetailSections[detailExpandKey(suggestionKey, 'occasion')])
                const isDetailsExpanded = Boolean(
                  expandedDetailSections[detailExpandKey(suggestionKey, 'details')],
                )
                const highlightTerms = getHighlightTerms(suggestion.matched_keywords.map((match) => match.keyword))
                const hasExpandableDetails =
                  Boolean(suggestion.latent_radar_chart) || suggestion.matched_keywords.length > 0

                return (
                  <article
                    key={suggestionKey}
                    className={`suggestion-card ${index === 0 ? 'is-top-choice' : ''}`}
                  >
                    <div className="card-hero">
                      <div className="card-hero-text">
                        <div className="card-topline">
                          <span className="rank-badge">#{index + 1}</span>
                        </div>
                        <div className="card-header">
                          <h2>{displayName}</h2>
                          <p>{suggestion.scientific_name}</p>
                        </div>
                        <div className="score-meter">
                          <div className="score-meter-bar">
                            <div className="score-track">
                              <div
                                className="score-fill"
                                data-strength={strength.toFixed(3)}
                                style={{ transform: `scaleX(${strength.toFixed(3)})` }}
                              />
                            </div>
                          </div>
                          <div className="score-caption">
                            <span>Match strength</span>
                            <strong>{strengthLabel}</strong>
                          </div>
                        </div>
                      </div>
                      {suggestion.image_url && (
                        <div className="card-hero-image">
                          <img className="suggestion-image" src={suggestion.image_url} alt={displayName} />
                        </div>
                      )}
                    </div>

                    <div className="card-metadata">
                      <div className="card-metadata-attrs" aria-label="Flower attributes">
                        <div className="detail-card detail-card--compact">
                          <span className="detail-label">Colors</span>
                          <p>{formatLabel(suggestion.colors)}</p>
                        </div>
                        <div className="detail-card detail-card--compact">
                          <span className="detail-label">Plant type</span>
                          <p>{formatLabel(suggestion.plant_types)}</p>
                        </div>
                        <div className="detail-card detail-card--compact">
                          <span className="detail-label">Maintenance</span>
                          <p>{formatMaintenanceLabel(suggestion.maintenance)}</p>
                        </div>
                      </div>
                      <div className="card-metadata-narrative" aria-label="Meaning and occasions">
                        <div className="detail-card detail-card--meaning">
                          <span className="detail-label">Meaning</span>
                          <div className="detail-copy-group">
                            <p
                              className={`detail-copy ${
                                meaningIsExpandable && !isMeaningExpanded ? 'is-collapsed' : ''
                              }`}
                            >
                              {renderHighlightedText(fullMeaningText, highlightTerms)}
                            </p>
                            {meaningIsExpandable && (
                              <button
                                type="button"
                                className="detail-toggle"
                                onClick={() =>
                                  setExpandedDetailSections((currentState) => {
                                    const key = detailExpandKey(suggestionKey, 'meaning')
                                    return { ...currentState, [key]: !currentState[key] }
                                  })
                                }
                              >
                                {isMeaningExpanded ? 'show less' : 'show full meaning'}
                              </button>
                            )}
                          </div>
                        </div>
                        {suggestion.occasions && suggestion.occasions.length > 0 && (
                          <div className="detail-card detail-card--occasions">
                            <span className="detail-label">Occasions</span>
                            <div className="detail-copy-group">
                              <p
                                className={`detail-copy ${
                                  occasionIsExpandable && !isOccasionExpanded ? 'is-collapsed' : ''
                                }`}
                              >
                                {renderHighlightedText(fullOccasionText, highlightTerms)}
                              </p>
                              {occasionIsExpandable && (
                                <button
                                  type="button"
                                  className="detail-toggle"
                                  onClick={() =>
                                    setExpandedDetailSections((currentState) => {
                                      const key = detailExpandKey(suggestionKey, 'occasion')
                                      return { ...currentState, [key]: !currentState[key] }
                                    })
                                  }
                                >
                                  {isOccasionExpanded ? 'show less' : 'show full occasions'}
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {hasExpandableDetails && (
                      <div className="card-details-drawer">
                        <button
                          type="button"
                          className="card-details-toggle"
                          aria-expanded={isDetailsExpanded}
                          onClick={() =>
                            setExpandedDetailSections((currentState) => {
                              const key = detailExpandKey(suggestionKey, 'details')
                              return { ...currentState, [key]: !currentState[key] }
                            })
                          }
                        >
                          {isDetailsExpanded ? 'Hide Details - ' : 'Expand Details +'}
                        </button>
                        {isDetailsExpanded && (
                          <div className="card-details-body">
                            {suggestion.latent_radar_chart && (
                              <figure className="latent-radar-panel latent-radar-panel--inline">
                                <div className="latent-radar-head">
                                </div>
                                <img
                                  src={suggestion.latent_radar_chart}
                                  alt={`latent svd radar chart for ${displayName.toLowerCase()}. axes: ${(suggestion.latent_radar_axes ?? []).join(', ')}`}
                                />
                                <figcaption>{displayName.toLowerCase()} latent profile</figcaption>
                              </figure>
                            )}
                            {suggestion.matched_keywords.length > 0 && (
                              <div className="match-list match-list--in-details">
                                {suggestion.matched_keywords.map((match, matchIndex) => (
                                  <div key={`${match.keyword}-${match.category}-${matchIndex}`} className="match-chip">
                                    <span>{match.keyword}</span>
                                    <small>{match.category}</small>
                                    <strong>+{match.score.toFixed(2)}</strong>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </article>
                )
              })}
            </div>
          ) : (
            <div className="results-empty">
              No suggestions yet. Start with a color, flower meaning, or maintenance level.
            </div>
          )}
        </div>
        </section>
      </div>
    </main>
  )
}

export default App
