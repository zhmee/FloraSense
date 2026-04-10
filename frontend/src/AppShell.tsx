/**
 * AppShell — Top-level layout wrapper.
 */
import { useRef, useState, useCallback } from 'react'
import HeroSection from './HeroSection'
import StickyNav, { ActiveView } from './StickyNav'
import App from './App'
import Visualizer3D from './Visualizer3D'
import './AppShell.css'

function AppShell(): JSX.Element {
  const [activeView, setActiveView] = useState<ActiveView>('ranker')
  const contentRef = useRef<HTMLDivElement>(null)
  const isVisualizerView = activeView === 'visualizer'

  const scrollToContent = useCallback(() => {
    contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const handleViewChange = useCallback((view: ActiveView) => {
    setActiveView(view)
    window.setTimeout(() => {
      if (view === 'visualizer') {
        window.scrollTo({ top: 0, behavior: 'smooth' })
        return
      }
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
          <Visualizer3D isActive={activeView === 'visualizer'} />
        </div>
      </div>
    </div>
  )
}

export default AppShell
