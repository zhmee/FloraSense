/**
 * AppShell — Top-level layout wrapper.
 */
import { Suspense, lazy, useRef, useState, useCallback, useEffect } from 'react'
import HeroSection from './HeroSection'
import StickyNav, { ActiveView } from './StickyNav'
import App from './App'
import './AppShell.css'

const Visualizer3D = lazy(() => import('./Visualizer3D'))

function AppShell(): JSX.Element {
  const [activeView, setActiveView] = useState<ActiveView>('ranker')
  const contentRef = useRef<HTMLDivElement>(null)
  const isVisualizerView = activeView === 'visualizer'

  useEffect(() => {
    if (!isVisualizerView) return

    const previousBodyOverflow = document.body.style.overflow
    const previousHtmlOverflow = document.documentElement.style.overflow
    document.body.style.overflow = 'hidden'
    document.documentElement.style.overflow = 'hidden'

    return () => {
      document.body.style.overflow = previousBodyOverflow
      document.documentElement.style.overflow = previousHtmlOverflow
    }
  }, [isVisualizerView])

  const scrollToContent = useCallback(() => {
    contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const handleViewChange = useCallback((view: ActiveView) => {
    setActiveView(view)
    if (view === 'visualizer') {
      // Avoid an initial "jump then smooth scroll up" flash when switching views.
      window.scrollTo({ top: 0, behavior: 'auto' })
      return
    }
    window.setTimeout(() => {
      contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 60)
  }, [])

  return (
    <div className={`fs-shell ${isVisualizerView ? 'fs-shell--visualizer' : 'fs-shell--ranker'}`}>
      {!isVisualizerView ? (
        <div className="fs-shell__hero">
          <HeroSection onScrollDown={scrollToContent} />
        </div>
      ) : null}

      <StickyNav
        visible
        activeView={activeView}
        onViewChange={handleViewChange}
      />

      <div
        ref={contentRef}
        className="fs-shell__content"
      >
        <div
          className={`fs-shell__panel fs-shell__panel--ranker ${activeView === 'ranker' ? 'fs-shell__panel--active' : 'fs-shell__panel--inactive'}`}
          aria-hidden={activeView !== 'ranker'}
        >
          <App isActive={activeView === 'ranker'} />
        </div>
        <div
          className={`fs-shell__panel fs-shell__panel--visualizer ${activeView === 'visualizer' ? 'fs-shell__panel--active' : 'fs-shell__panel--inactive'}`}
          aria-hidden={activeView !== 'visualizer'}
        >
          <Suspense fallback={<div className="fs-shell__loading">Loading visualizer...</div>}>
            <Visualizer3D isActive={activeView === 'visualizer'} />
          </Suspense>
        </div>
      </div>
    </div>
  )
}

export default AppShell
