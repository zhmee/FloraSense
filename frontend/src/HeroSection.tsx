/**
 * HeroSection — Full-screen animated landing section for FloraSense.
 * Uses Anime.js for the intro reveal and subtle proximity bloom on hero flowers.
 */
import { useEffect, useRef } from 'react'
import { animate, stagger, createTimeline } from 'animejs'
import FlowerCoral from './assets/flower-coral.svg'
import FlowerGold from './assets/flower-gold.svg'
import FlowerOlive from './assets/flower-olive.svg'
import FlowerRose from './assets/flower-rose.svg'
import './HeroSection.css'

const PETAL_FLOWERS = [
  { src: FlowerCoral, cls: 'hero-petal hero-petal--coral', depth: 1.28, style: { top: '8%', left: '6%', width: 160 } },
  { src: FlowerGold, cls: 'hero-petal hero-petal--gold', depth: 1.16, style: { top: '12%', right: '8%', width: 130 } },
  { src: FlowerOlive, cls: 'hero-petal hero-petal--olive', depth: 1.22, style: { bottom: '18%', left: '10%', width: 120 } },
  { src: FlowerRose, cls: 'hero-petal hero-petal--rose', depth: 1.32, style: { bottom: '22%', right: '6%', width: 150 } },
  { src: FlowerCoral, cls: 'hero-petal hero-petal--coral2', depth: 1.02, style: { top: '42%', left: '2%', width: 90 } },
  { src: FlowerGold, cls: 'hero-petal hero-petal--gold2', depth: 1.08, style: { top: '55%', right: '3%', width: 100 } },
  { src: FlowerOlive, cls: 'hero-petal hero-petal--olive2', depth: 0.92, style: { top: '22%', left: '22%', width: 70 } },
  { src: FlowerRose, cls: 'hero-petal hero-petal--rose2', depth: 0.96, style: { top: '18%', right: '22%', width: 75 } },
  { src: FlowerCoral, cls: 'hero-petal hero-petal--xs', depth: 0.84, style: { bottom: '38%', left: '28%', width: 52 } },
  { src: FlowerGold, cls: 'hero-petal hero-petal--xs2', depth: 0.8, style: { bottom: '42%', right: '26%', width: 48 } },
  { src: FlowerRose, cls: 'hero-petal hero-petal--xs3', depth: 0.86, style: { top: '68%', left: '18%', width: 58 } },
  { src: FlowerOlive, cls: 'hero-petal hero-petal--xs4', depth: 0.84, style: { top: '72%', right: '16%', width: 54 } },
]

interface HeroSectionProps {
  onScrollDown: () => void
}

function HeroSection({ onScrollDown }: HeroSectionProps): JSX.Element {
  const heroRef = useRef<HTMLElement>(null)
  const hasAnimated = useRef(false)

  useEffect(() => {
    const hero = heroRef.current
    if (!hero) return

    const petals = Array.from(hero.querySelectorAll<HTMLElement>('.hero-petal'))
    const eyebrow = hero.querySelector<HTMLElement>('.hero-eyebrow')
    const titleCharacters = Array.from(hero.querySelectorAll<HTMLElement>('.hero-title-char'))
    const tagline = hero.querySelector<HTMLElement>('.hero-tagline')
    const cta = hero.querySelector<HTMLElement>('.hero-scroll-cta')
    let introFrame = 0

    const revealHero = () => {
      hero.dataset.ready = 'true'
      petals.forEach((petal) => {
        petal.style.opacity = '1'
      })
      eyebrow?.style.setProperty('opacity', '1')
      titleCharacters.forEach((character) => {
        character.style.opacity = '1'
        character.style.transform = 'translateY(0) rotate(0deg) scale(1)'
      })
      tagline?.style.setProperty('opacity', '1')
      cta?.style.setProperty('opacity', '1')
    }

    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReduced) {
      hasAnimated.current = true
      revealHero()
      return
    }

    introFrame = window.requestAnimationFrame(() => {
      hero.dataset.ready = 'true'

      if (!hasAnimated.current) {
        hasAnimated.current = true

        animate(petals, {
          opacity: [0, 1],
          scale: [0.3, 1],
          rotate: () => `${(Math.random() - 0.5) * 40}deg`,
          delay: stagger(90, { from: 'center' }),
          duration: 1100,
          ease: 'spring(1, 80, 12, 0)',
        })

        const tl = createTimeline({ defaults: { ease: 'out(4)' } })

        if (eyebrow) {
          tl.add(eyebrow, {
            opacity: [0, 1],
            translateY: [24, 0],
            duration: 600,
          }, 400)
        }

        if (titleCharacters.length > 0) {
          tl.add(titleCharacters, {
            opacity: [0, 1],
            translateY: [60, 0],
            rotate: ['-8deg', '0deg'],
            scale: [0.7, 1],
            delay: stagger(55),
            duration: 700,
            ease: 'spring(1, 90, 10, 0)',
          }, 700)
        }

        if (tagline) {
          tl.add(tagline, {
            opacity: [0, 1],
            translateY: [18, 0],
            duration: 550,
          }, 1100)
        }

        if (cta) {
          tl.add(cta, {
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 500,
          }, 1400)
        }
      } else {
        revealHero()
      }
    })

    const states = petals.map(() => ({
      x: 0,
      y: 0,
      scale: 1,
      brightness: 1,
      saturation: 1,
      shadowY: 12,
      shadowBlur: 24,
      shadowAlpha: 0.14,
      phase: Math.random() * Math.PI * 2,
      speed: 0.00045 + Math.random() * 0.00035,
      amplitudeX: 8 + Math.random() * 16,
      amplitudeY: 10 + Math.random() * 18,
      spinAmplitude: 3 + Math.random() * 8,
      rotationBoost: 0,
    }))
    let pointerX = window.innerWidth * 0.5
    let pointerY = window.innerHeight * 0.5
    let frameId = 0

    const render = (time: number) => {
      petals.forEach((petal, index) => {
        const state = states[index]
        const angle = time * state.speed + state.phase
        const ambientX = Math.sin(angle) * state.amplitudeX
        const ambientY = Math.cos(angle * 0.72) * state.amplitudeY
        const rect = petal.getBoundingClientRect()
        const centerX = rect.left + rect.width / 2
        const centerY = rect.top + rect.height / 2
        const dx = centerX - pointerX
        const dy = centerY - pointerY
        const distance = Math.hypot(dx, dy) || 1
        const radius = rect.width * 0.95 + 90
        const strength = Math.max(0, 1 - distance / radius)
        const depth = Number(petal.dataset.depth ?? '1')

        const desiredX = strength > 0 ? (dx / distance) * strength * 38 * depth : 0
        const desiredY = strength > 0 ? (dy / distance) * strength * 32 * depth : 0
        const desiredScale = 1 + strength * 0.28
        const desiredBrightness = 1 + strength * 0.22
        const desiredSaturation = 1 + strength * 0.9
        const desiredShadowY = 12 + strength * 18
        const desiredShadowBlur = 24 + strength * 38
        const desiredShadowAlpha = 0.14 + strength * 0.28
        const desiredRotationBoost = strength * 15

        state.x += (desiredX - state.x) * 0.18
        state.y += (desiredY - state.y) * 0.18
        state.scale += (desiredScale - state.scale) * 0.18
        state.brightness += (desiredBrightness - state.brightness) * 0.18
        state.saturation += (desiredSaturation - state.saturation) * 0.18
        state.shadowY += (desiredShadowY - state.shadowY) * 0.18
        state.shadowBlur += (desiredShadowBlur - state.shadowBlur) * 0.18
        state.shadowAlpha += (desiredShadowAlpha - state.shadowAlpha) * 0.18
        state.rotationBoost += (desiredRotationBoost - state.rotationBoost) * 0.18

        const rotation = Math.sin(angle * 0.5) * state.spinAmplitude + state.rotationBoost
        petal.style.transform =
          `translate3d(${ambientX + state.x}px, ${ambientY + state.y}px, 0) rotate(${rotation}deg) scale(${state.scale})`
        petal.style.filter =
          `drop-shadow(0 ${state.shadowY}px ${state.shadowBlur}px rgba(109, 81, 47, ${state.shadowAlpha})) ` +
          `drop-shadow(0 0 ${16 + strength * 28}px rgba(255, 239, 196, ${0.08 + strength * 0.24})) ` +
          `saturate(${state.saturation}) brightness(${state.brightness})`
      })

      frameId = window.requestAnimationFrame(render)
    }

    const onPointerMove = (event: PointerEvent) => {
      pointerX = event.clientX
      pointerY = event.clientY
    }

    const onPointerLeave = () => {
      pointerX = -1000
      pointerY = -1000
    }

    frameId = window.requestAnimationFrame(render)
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerleave', onPointerLeave)

    return () => {
      window.cancelAnimationFrame(introFrame)
      window.cancelAnimationFrame(frameId)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerleave', onPointerLeave)
    }
  }, [])

  const titleChars = 'FloraSense'.split('')

  return (
    <section ref={heroRef} className="fs-hero" data-ready="false" aria-label="FloraSense welcome">
      <div className="fs-hero__petals" aria-hidden="true">
        {PETAL_FLOWERS.map((p, i) => (
          <img
            key={i}
            src={p.src}
            alt=""
            className={p.cls}
            data-depth={p.depth}
            style={{
              position: 'absolute',
              height: p.style.width,
              opacity: 0,
              ...p.style,
            }}
          />
        ))}
      </div>

      <div className="fs-hero__orb fs-hero__orb--gold" aria-hidden="true" />
      <div className="fs-hero__orb fs-hero__orb--rose" aria-hidden="true" />
      <div className="fs-hero__orb fs-hero__orb--olive" aria-hidden="true" />

      <div className="fs-hero__content">
        <p className="hero-eyebrow fs-hero__eyebrow" style={{ opacity: 0 }}>
          Describe a feeling. Discover a flower.
        </p>

        <h1 className="fs-hero__title" aria-label="FloraSense">
          {titleChars.map((ch, i) => (
            <span
              key={i}
              className="hero-title-char fs-hero__title-char"
              style={{ opacity: 0, display: 'inline-block' }}
              aria-hidden="true"
            >
              {ch}
            </span>
          ))}
        </h1>

        <button
          className="hero-scroll-cta fs-hero__cta"
          style={{ opacity: 0 }}
          onClick={onScrollDown}
          aria-label="Scroll to the app"
        >
          <span>Explore the Garden</span>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path d="M10 4v12M4 10l6 6 6-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </section>
  )
}

export default HeroSection
