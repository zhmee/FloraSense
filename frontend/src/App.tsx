import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from 'react'
import { animate, createScope, stagger } from 'animejs'
import type { Target } from 'animejs'
import './App.css'
import SearchIcon from './assets/mag.png'
import FlowerCoral from './assets/flower-coral.svg'
import FlowerGold from './assets/flower-gold.svg'
import FlowerOlive from './assets/flower-olive.svg'
import FlowerRose from './assets/flower-rose.svg'
import { AutocompleteResponse, RecommendationResponse } from './types'

const EMPTY_RESULTS: RecommendationResponse = {
  query: '',
  keywords_used: [],
  suggestions: [],
}

const SAMPLE_QUERIES = [
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

function formatLabel(values: string[]): string {
  return values.length > 0 ? values.join(', ') : 'Not listed'
}

function formatFullText(values: string[]): string {
  if (values.length === 0) return 'Not listed'
  const combined = values.join(' ').replace(/\s+/g, ' ').trim()
  return combined || 'Not listed'
}

function formatLongText(values: string[], maxSentences: number = 2): string {
  if (values.length === 0) return 'Not listed'

  const combined = values.join(' ').replace(/\s+/g, ' ').trim()
  if (!combined) return 'Not listed'

  const sentences = combined.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [combined]
  return sentences
    .slice(0, maxSentences)
    .map((sentence) => sentence.trim())
    .join(' ')
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function getHighlightTerms(values: string[]): string[] {
  const deduped = Array.from(
    new Set(
      values
        .map((value) => value.trim().toLowerCase())
        .filter((value) => value.length >= 2),
    ),
  )

  return deduped.sort((left, right) => right.length - left.length)
}

function renderHighlightedText(text: string, highlightTerms: string[]): ReactNode {
  if (!text || text === 'Not listed' || highlightTerms.length === 0) {
    return text
  }

  const pattern = new RegExp(`(${highlightTerms.map(escapeRegExp).join('|')})`, 'gi')
  const parts = text.split(pattern)

  return parts.map((part, index) => {
    const isMatch = highlightTerms.some((term) => term === part.toLowerCase())
    return isMatch ? (
      <mark key={`${part}-${index}`} className="detail-highlight">
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    )
  })
}

function getMatchStrength(score: number, bestScore: number): number {
  if (bestScore <= 0) return 0
  return Math.max(Math.min(score / bestScore, 1), 0.12)
}

function App(): JSX.Element {
  const [query, setQuery] = useState<string>('')
  const [results, setResults] = useState<RecommendationResponse>(EMPTY_RESULTS)
  const [autocompleteSuggestions, setAutocompleteSuggestions] = useState<string[]>([])
  const [autocompleteOpen, setAutocompleteOpen] = useState<boolean>(false)
  const [autocompleteLoading, setAutocompleteLoading] = useState<boolean>(false)
  const [autocompleteEnabled, setAutocompleteEnabled] = useState<boolean>(false)
  const [activeAutocompleteIndex, setActiveAutocompleteIndex] = useState<number>(-1)
  const [expandedMeaningCards, setExpandedMeaningCards] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const queryInputRef = useRef<HTMLInputElement | null>(null)
  const appRef = useRef<HTMLElement | null>(null)
  const animationScopeRef = useRef<ReturnType<typeof createScope> | null>(null)
  const loadingAnimationRef = useRef<ReturnType<typeof animate> | null>(null)
  const autocompleteRequestRef = useRef<number>(0)
  const skipNextAutocompleteRef = useRef<boolean>(false)

  const topScore = results.suggestions[0]?.score ?? 0

  useEffect(() => {
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
        scaleX: (target: Target) => Number((target as HTMLElement).dataset.strength ?? '0'),
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
  }, [])

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

    if (results.keywords_used.length > 0) {
      scope.methods.animateKeywords?.()
    }

    if (results.suggestions.length > 0) {
      scope.methods.animateSuggestions?.()
    }
  }, [results, loading])

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
    setExpandedMeaningCards({})

    if (!trimmedQuery) {
      setResults(EMPTY_RESULTS)
      setError('')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(`/api/recommendations?q=${encodeURIComponent(trimmedQuery)}`)
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data: RecommendationResponse = await response.json()
      setResults(data)
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

      <section className="hero">
        <div className="hero-copy-block">
          <p className="eyebrow">Designed by feeling, defined by flowers, </p>
          <h1>FloraSense</h1>
          <p className="hero-copy">
            Describe a mood, color, level of care, or meaning, and receive a short ranked set of flower
            suggestions that feels more curated than filtered.
          </p>

          <form className="search-panel" onSubmit={handleSubmit}>
            <div className="search-row">
              <img src={SearchIcon} alt="" aria-hidden="true" />
              <input
                ref={queryInputRef}
                id="flower-query"
                value={query}
                onChange={(event) => handleQueryChange(event.target.value)}
                onKeyDown={handleQueryKeyDown}
                onFocus={() => {
                  setAutocompleteEnabled(true)
                  if (autocompleteSuggestions.length > 0) {
                    setAutocompleteOpen(true)
                  }
                }}
                onBlur={() => {
                  window.setTimeout(() => setAutocompleteOpen(false), 120)
                }}
                placeholder="e.g. a low maintenance white flower for love"
                autoComplete="off"
                aria-autocomplete="list"
                aria-expanded={autocompleteOpen}
                aria-controls="flower-autocomplete-list"
              />
              <button type="submit" disabled={loading}>
                {loading ? 'Ranking...' : 'Search'}
              </button>
            </div>
          </form>

          <div
            className={`sample-row ${(autocompleteOpen || autocompleteLoading) ? 'is-autocomplete' : ''}`}
            id="flower-autocomplete-list"
            role={(autocompleteOpen || autocompleteLoading) ? 'listbox' : undefined}
          >
            <div className={`sample-row-content ${(autocompleteOpen || autocompleteLoading) ? 'is-hidden' : ''}`}>
              {SAMPLE_QUERIES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="sample-chip"
                  onClick={() => void runSearch(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
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
                    <small>SVD suggestion</small>
                  </button>
                ))}
                {autocompleteLoading && autocompleteSuggestions.length === 0 && (
                  <div className="autocomplete-empty">Loading suggestions...</div>
                )}
              </>
            </div>
          </div>
        </div>

        <div className="hero-side">
          <div className="hero-note">
            <span>How it works:</span>
            <p>
              The recommender projects flowers and queries into an SVD-based retrieval space built
              from meanings, attributes, and flower descriptions, then ranks the closest matches.
            </p>
          </div>
        </div>
      </section>

      <section className="page-body">
        <aside className="query-rail">
          <div className="rail-heading">
            <p>Query Breakdown</p>
            <span>{results.keywords_used.length}</span>
          </div>
          <p className="rail-copy">
            Highest-weight semantic terms from the current request:
          </p>

          {loading ? (
            <div className="rail-empty">
              Projecting the query into the semantic flower space.
            </div>
          ) : results.keywords_used.length > 0 ? (
            <div className="keyword-grid">
              {results.keywords_used.map((keyword) => (
                <div key={`${keyword.category}-${keyword.keyword}`} className="keyword-card">
                  <strong>{keyword.keyword}</strong>
                  <span>{keyword.category}</span>
                  <span className="score-pill">+{keyword.score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="rail-empty">
              {error || 'Run a search to see which semantic terms influenced the ranking.'}
            </div>
          )}
        </aside>

        <div className="results-column">
          <header className="results-header">
            <div className="results-title-block">
              <h2>Best matches for...</h2>
              {results.query && !loading && <p>“{results.query}”</p>}
            </div>
            <div className="results-meta">
              <span>{results.suggestions.length}/5 shown</span>
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
                const strength = getMatchStrength(suggestion.score, topScore)
                const strengthLabel = Math.round(strength * 100)
                const suggestionKey = `${suggestion.name}-${suggestion.scientific_name}`
                const fullMeaningText = formatFullText(suggestion.meanings)
                const fullOccasionText = formatLongText(suggestion.occasions)
                const meaningIsExpandable = fullMeaningText !== 'Not listed' && fullMeaningText.length > 220
                const isMeaningExpanded = Boolean(expandedMeaningCards[suggestionKey])
                const highlightTerms = getHighlightTerms(suggestion.matched_keywords.map((match) => match.keyword))

                return (
                  <article
                    key={suggestionKey}
                    className={`suggestion-card ${index === 0 ? 'is-top-choice' : ''}`}
                  >
                    {index === 0 && <span className="top-pick-banner">Top recommendation</span>}

                    <div className="card-topline">
                      <span className="rank-badge">#{index + 1}</span>
                      <span className="score-badge">{suggestion.score.toFixed(2)}</span>
                    </div>

                    <div className="card-header">
                      <h2>{suggestion.name}</h2>
                      <p>{suggestion.scientific_name}</p>
                    </div>

                    <div className="score-meter">
                      <div className="score-track">
                        <div
                          className="score-fill"
                          data-strength={strength.toFixed(3)}
                          style={{ transform: 'scaleX(0)' }}
                        />
                      </div>
                      <div className="score-caption">
                        <span>Match strength</span>
                        <strong>{strengthLabel}%</strong>
                      </div>
                    </div>

                    <div className="detail-grid">
                      <div>
                        <span className="detail-label">Colors</span>
                        <p>{formatLabel(suggestion.colors)}</p>
                      </div>
                      <div>
                        <span className="detail-label">Maintenance</span>
                        <p>{formatLabel(suggestion.maintenance)}</p>
                      </div>
                      <div>
                        <span className="detail-label">Plant Type</span>
                        <p>{formatLabel(suggestion.plant_types)}</p>
                      </div>
                      <div>
                        <span className="detail-label">Meaning</span>
                        <div className="detail-copy-group">
                          <p className={`detail-copy ${meaningIsExpandable && !isMeaningExpanded ? 'is-collapsed' : ''}`}>
                            {renderHighlightedText(fullMeaningText, highlightTerms)}
                          </p>
                          {meaningIsExpandable && (
                            <button
                              type="button"
                              className="detail-toggle"
                              onClick={() =>
                                setExpandedMeaningCards((currentState) => ({
                                  ...currentState,
                                  [suggestionKey]: !currentState[suggestionKey],
                                }))
                              }
                            >
                              {isMeaningExpanded ? 'show less' : 'show full meaning'}
                            </button>
                          )}
                        </div>
                      </div>
                      {suggestion.occasions && (
                        <div>
                          <span className="detail-label">Occasions</span>
                          <p>{renderHighlightedText(fullOccasionText, highlightTerms)}</p>
                        </div>
                      )}
                    </div>

                    <div className="match-list">
                      {suggestion.matched_keywords.map((match, matchIndex) => (
                        <div key={`${match.keyword}-${match.category}-${matchIndex}`} className="match-chip">
                          <span>{match.keyword}</span>
                          <small>{match.category}</small>
                          <strong>+{match.score.toFixed(2)}</strong>
                        </div>
                      ))}
                    </div>
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
    </main>
  )
}

export default App
