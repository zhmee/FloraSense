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

  const scrollToContent = useCallback(() => {
    contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const handleViewChange = useCallback((view: ActiveView) => {
    setActiveView(view)
    window.setTimeout(() => {
      contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 60)
  }, [])

  return (
    <div className="fs-shell">
      {/* 1. Hero */}
      <div>
        <HeroSection onScrollDown={scrollToContent} />
      </div>

      {/* 2. Sticky nav */}
      <StickyNav
        visible
        activeView={activeView}
        onViewChange={handleViewChange}
      />

      {/* 3. Content — nav-offset spacer only appears when nav is sticky */}
      <div
        ref={contentRef}
        className="fs-shell__content"
      >
        {/* Render both, hide inactive to preserve existing App state */}
        <div style={{ display: activeView === 'ranker' ? 'block' : 'none' }}>
          <App />
        </div>
        <div style={{ display: activeView === 'visualizer' ? 'block' : 'none' }}>
          <Visualizer3D />
        </div>
      </div>
    </div>
  )
}

export default AppShell
