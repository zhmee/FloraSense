/**
 * StickyNav — appears once user scrolls past hero section.
 * Provides tab switching between Ranker and the 3D visualizer.
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
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <rect x="2.25" y="9.5" width="2.5" height="5.25" rx="1.1" fill="currentColor" />
              <rect x="7.1" y="6.25" width="2.5" height="8.5" rx="1.1" fill="currentColor" opacity="0.92" />
              <rect x="11.95" y="3.25" width="2.5" height="11.5" rx="1.1" fill="currentColor" opacity="0.82" />
              <path
                d="M2.7 5.9C4.1 5.2 5.15 4.85 6.05 4.85C7.2 4.85 7.8 5.7 8.9 5.7C10.15 5.7 11.05 4.1 13.65 3.55"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Ranker
          </button>
          <button
            role="tab"
            aria-selected={activeView === 'visualizer'}
            className={`fs-nav__tab ${activeView === 'visualizer' ? 'fs-nav__tab--active' : ''}`}
            onClick={() => onViewChange('visualizer')}
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <path
                d="M9 2.4L14.5 5.55V12.45L9 15.6L3.5 12.45V5.55L9 2.4Z"
                stroke="currentColor"
                strokeWidth="1.35"
                strokeLinejoin="round"
              />
              <path d="M9 2.4V9M14.5 5.55L9 9L3.5 5.55" stroke="currentColor" strokeWidth="1.35" strokeLinejoin="round" />
              <circle cx="9" cy="9" r="1.55" fill="currentColor" />
            </svg>
            3D Visualizer
          </button>
        </div>
      </div>
    </nav>
  )
}

export default StickyNav
