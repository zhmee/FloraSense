/**
 * StickyNav — appears once user scrolls past hero section.
 * Provides tab switching between Ranker and 3D Visualizer.
 */
import { useEffect, useRef } from 'react'
import { animate } from 'animejs'
import './StickyNav.css'

export type ActiveView = 'ranker' | 'visualizer'

interface StickyNavProps {
  visible: boolean
  activeView: ActiveView
  onViewChange: (view: ActiveView) => void
}

function StickyNav({ visible, activeView, onViewChange }: StickyNavProps): JSX.Element {
  const navRef = useRef<HTMLElement>(null)
  const wasVisible = useRef(false)
  const hasAnimatedIn = useRef(false)

  useEffect(() => {
    if (!navRef.current) return
    if (visible === wasVisible.current) return
    wasVisible.current = visible

    if (visible) {
      animate(navRef.current, {
        opacity: [0, 1],
        translateY: [-20, 0],
        delay: hasAnimatedIn.current ? 0 : 1500,
        duration: 380,
        ease: 'out(4)',
      })
      hasAnimatedIn.current = true
    }
  }, [visible])

  return (
    <nav
      ref={navRef}
      className={`fs-nav ${visible ? 'fs-nav--visible' : 'fs-nav--hidden'}`}
      aria-label="Main navigation"
      style={{ opacity: 0 }}
    >
      <div className="fs-nav__inner">
        <div className="fs-nav__brand">
          <span className="fs-nav__logo">✿</span>
          <span className="fs-nav__wordmark">FloraSense</span>
        </div>

        <div className="fs-nav__tabs" role="tablist">
          <button
            role="tab"
            aria-selected={activeView === 'ranker'}
            className={`fs-nav__tab ${activeView === 'ranker' ? 'fs-nav__tab--active' : ''}`}
            onClick={() => onViewChange('ranker')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M2 13h2V6H2v7zm3-5h2V3H5v5zm3 3h2V8H8v3zm3-6h2v6h-2V5z" fill="currentColor"/>
            </svg>
            Ranker
          </button>
          <button
            role="tab"
            aria-selected={activeView === 'visualizer'}
            className={`fs-nav__tab ${activeView === 'visualizer' ? 'fs-nav__tab--active' : ''}`}
            onClick={() => onViewChange('visualizer')}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="8" cy="8" r="3" fill="currentColor" opacity="0.9"/>
              <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.1"/>
              <line x1="8" y1="1.5" x2="8" y2="3.5" stroke="currentColor" strokeWidth="1.2"/>
              <line x1="8" y1="12.5" x2="8" y2="14.5" stroke="currentColor" strokeWidth="1.2"/>
              <line x1="1.5" y1="8" x2="3.5" y2="8" stroke="currentColor" strokeWidth="1.2"/>
              <line x1="12.5" y1="8" x2="14.5" y2="8" stroke="currentColor" strokeWidth="1.2"/>
            </svg>
            3D Visualizer
          </button>
        </div>
      </div>
    </nav>
  )
}

export default StickyNav
