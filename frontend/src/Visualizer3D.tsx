import { animate, stagger } from 'animejs'
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { BouquetInsightsResponse, VisualizerFlower, VisualizerFlowersResponse } from './types'
import './Visualizer3D.css'

const TAU = Math.PI * 2
const MAX_NEIGHBORS_PER_FLOWER = 2
const CAMERA_FOV = 980
const NODE_DEPTH_SPREAD = 360
const DEFAULT_ORBIT_YAW = -0.18
const DEFAULT_ORBIT_PITCH = 0.22
const MIN_ORBIT_PITCH = -0.12
const MAX_ORBIT_PITCH = 0.7

type SceneStatus = 'loading' | 'ready' | 'error'
type EdgeReason = 'color' | 'meaning' | 'occasion' | 'semantic'
type InsightsStatus = 'idle' | 'loading' | 'ready' | 'error'

interface Point2 { x: number; y: number }
interface Point3 { x: number; y: number; z: number }

interface SemanticEdge {
  key: string; sourceId: string; targetId: string
  similarity: number; reason: EdgeReason
  color: string; width: number; baseOpacity: number; label: string | null; score: number
}
interface SemanticGraph { edges: SemanticEdge[]; neighborMap: Map<string, SemanticEdge[]> }

interface Bouquet {
  id: string; label: string; descriptor: string; color: string
  meaning: string; occasion: string; primaryColor: string
  centroidFlowerId: string; importance: number
  memberIds: string[]; previewIds: string[]
}

interface FlowerNode {
  data: VisualizerFlower
  pos: Point2; targetPos: Point2
  scale: number; targetScale: number
  glow: number; targetGlow: number
  petalCount: number; petalAngleOffset: number
  spinAngle: number; driftPhase: number
  importance: number; bouquetId: string | null
  worldZ: number
}

interface BouquetAnchor {
  data: Bouquet
  pos: Point2; targetPos: Point2
  scale: number; targetScale: number
  glow: number; targetGlow: number
  ringAngle: number
}

interface EdgeLink {
  data: SemanticEdge; opacity: number; targetOpacity: number
}

interface ProjectionCamera {
  x: number
  y: number
  z: number
  zoom: number
  yaw: number
  pitch: number
  width: number
  height: number
}

interface ProjectedNode {
  x: number
  y: number
  radius: number
  depth: number
  scale: number
  visible: boolean
}

interface Visualizer3DProps {
  isActive?: boolean
}

// ─── Color helpers ────────────────────────────────────────────────────────────

const COLOR_HEX_MAP: Record<string, string> = {
  red:'#da4b54',crimson:'#b83b4d',scarlet:'#c63e4a',
  pink:'#dc7fa6',blush:'#dc9ab1',coral:'#ea836d',
  orange:'#df8640',peach:'#f0af77',
  yellow:'#e3b53a',gold:'#d7a341',
  white:'#f5efe6',ivory:'#ede0d3',cream:'#f2e5cf',
  blue:'#5e86d9',azure:'#4e86d7',
  purple:'#8d72d5',lavender:'#9d8ce0',lilac:'#ac8ad7',violet:'#785ec7',
  green:'#74a56f',sage:'#8da680',peachpuff:'#efbb9a',burgundy:'#8f3448',
}

function normalizeKey(v: string) { return v.trim().toLowerCase() }
function toTitleCase(v: string) {
  return v.trim().split(/\s+/).filter(Boolean).map(p => p[0].toUpperCase() + p.slice(1)).join(' ')
}
function getColorHex(v: string) { return COLOR_HEX_MAP[normalizeKey(v)] ?? '#d9a774' }
function getFlowerColorHex(f: VisualizerFlower) {
  for (const c of f.colors) { const h = getColorHex(c); if (h !== '#d9a774') return h }
  return getColorHex(f.primary_color)
}
function clamp(v: number, lo: number, hi: number) { return Math.min(hi, Math.max(lo, v)) }
function lerp(a: number, b: number, t: number) { return a + (b - a) * t }
function toPercent(v: number) { return `${Math.round(clamp(v, 0, 1) * 100)}%` }
function toScoreLabel(v: number) { return `${Math.round(v)}% match` }

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}
function rgbToHex(r: number, g: number, b: number) {
  return '#' + [r, g, b].map(v => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('')
}
function darken(hex: string, f: number) {
  const [r, g, b] = hexToRgb(hex); return rgbToHex(r * f, g * f, b * f)
}
function lighten(hex: string, f: number) {
  const [r, g, b] = hexToRgb(hex)
  return rgbToHex(r + (255 - r) * f, g + (255 - g) * f, b + (255 - b) * f)
}
function withAlpha(hex: string, a: number) {
  const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`
}

function hashString(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

// ─── Flower painting ──────────────────────────────────────────────────────────

function getStamenColor(pr: number, pg: number, pb: number): string {
  return rgbToHex(Math.min(255, pr * 0.25 + 210), Math.min(255, pg * 0.25 + 155), Math.min(255, pb * 0.08 + 25))
}

function paintFlower(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, r: number,
  color: string, petalCount: number,
  spinAngle: number, glowStrength: number, selected: boolean,
) {
  const [pr, pg, pb] = hexToRgb(color)

  // outer glow / selection halo
  if (glowStrength > 0.01) {
    const gR = r * (selected ? 2.4 : 1.9)
    const g = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, gR)
    const gc = selected ? lighten(color, 0.3) : color
    g.addColorStop(0, withAlpha(gc, glowStrength * (selected ? 0.52 : 0.30)))
    g.addColorStop(1, withAlpha(gc, 0))
    ctx.save(); ctx.fillStyle = g
    ctx.beginPath(); ctx.arc(cx, cy, gR, 0, TAU); ctx.fill(); ctx.restore()
  }

  // soft drop shadow under petals
  ctx.save(); ctx.globalAlpha = 0.12
  for (let i = 0; i < petalCount; i++) {
    const angle = spinAngle + (i / petalCount) * TAU
    ctx.save()
    ctx.translate(cx + Math.cos(angle) * r * 0.42 + r * 0.06, cy + Math.sin(angle) * r * 0.42 + r * 0.08)
    ctx.rotate(angle + Math.PI / 2)
    _petalPath(ctx, r)
    ctx.fillStyle = darken(color, 0.22)
    ctx.fill(); ctx.restore()
  }
  ctx.restore()

  // back petals (offset half-step, slightly smaller, darker)
  for (let i = 0; i < petalCount; i += 2) {
    const angle = spinAngle + (i / petalCount) * TAU + (TAU / petalCount / 2)
    _paintPetal(ctx, cx, cy, angle, r * 0.88, darken(color, 0.78), lighten(darken(color, 0.78), 0.22), 0.82)
  }

  // front petals
  for (let i = 0; i < petalCount; i++) {
    const angle = spinAngle + (i / petalCount) * TAU
    _paintPetal(ctx, cx, cy, angle, r, color, lighten(color, 0.52), 1.0)
  }

  // veins
  ctx.save(); ctx.globalAlpha = 0.20
  ctx.strokeStyle = darken(color, 0.52)
  ctx.lineWidth = Math.max(0.4, r * 0.018)
  for (let i = 0; i < petalCount; i++) {
    const angle = spinAngle + (i / petalCount) * TAU
    const tx = cx + Math.cos(angle) * r * 0.94, ty = cy + Math.sin(angle) * r * 0.94
    ctx.beginPath()
    ctx.moveTo(cx + Math.cos(angle) * r * 0.13, cy + Math.sin(angle) * r * 0.13)
    ctx.lineTo(tx, ty); ctx.stroke()
    for (const sv of [-0.30, 0.30]) {
      const sa = angle + sv
      const sx2 = cx + Math.cos(sa) * r * 0.62, sy2 = cy + Math.sin(sa) * r * 0.62
      ctx.beginPath()
      ctx.moveTo(cx + Math.cos(angle) * r * 0.28, cy + Math.sin(angle) * r * 0.28)
      ctx.quadraticCurveTo(
        (cx + Math.cos(angle) * r * 0.38 + tx) * 0.5 + Math.cos(angle) * r * 0.06,
        (cy + Math.sin(angle) * r * 0.38 + ty) * 0.5 + Math.sin(angle) * r * 0.06,
        sx2, sy2
      ); ctx.stroke()
    }
  }
  ctx.restore()

  // center disc
  const cR = r * 0.27
  const cg = ctx.createRadialGradient(cx - cR * 0.25, cy - cR * 0.30, 0, cx, cy, cR)
  cg.addColorStop(0, lighten(darken(color, 0.52), 0.22))
  cg.addColorStop(0.55, darken(color, 0.52))
  cg.addColorStop(1, darken(color, 0.70))
  ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, cR, 0, TAU)
  ctx.fillStyle = cg; ctx.fill()
  ctx.globalAlpha = 0.45
  ctx.strokeStyle = darken(color, 0.62)
  ctx.lineWidth = Math.max(0.5, r * 0.022); ctx.stroke(); ctx.restore()

  // stamen cluster
  const sR = cR * 0.50
  const sc = getStamenColor(pr, pg, pb)
  const sg = ctx.createRadialGradient(cx, cy, 0, cx, cy, sR)
  sg.addColorStop(0, lighten(sc, 0.38)); sg.addColorStop(0.5, sc); sg.addColorStop(1, darken(sc, 0.65))
  ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, sR, 0, TAU); ctx.fillStyle = sg; ctx.fill(); ctx.restore()

  // stamen dots
  const dotCount = Math.min(petalCount, 10)
  const dotR = Math.max(1.0, r * 0.042)
  ctx.save()
  for (let i = 0; i < dotCount; i++) {
    const a = (i / dotCount) * TAU + spinAngle * 0.28
    const dx = cx + Math.cos(a) * sR * 1.42, dy = cy + Math.sin(a) * sR * 1.42
    const dg = ctx.createRadialGradient(dx - dotR * 0.3, dy - dotR * 0.3, 0, dx, dy, dotR)
    dg.addColorStop(0, lighten(sc, 0.55)); dg.addColorStop(1, darken(sc, 0.45))
    ctx.beginPath(); ctx.arc(dx, dy, dotR, 0, TAU); ctx.fillStyle = dg; ctx.fill()
  }
  ctx.restore()

  // specular on center
  const spg = ctx.createRadialGradient(cx - cR * 0.32, cy - cR * 0.38, 0, cx - cR * 0.1, cy - cR * 0.12, cR * 0.58)
  spg.addColorStop(0, 'rgba(255,255,255,0.40)'); spg.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, cR * 0.88, 0, TAU); ctx.fillStyle = spg; ctx.fill(); ctx.restore()
}

function paintFlowerImage(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  cx: number, cy: number, r: number,
  color: string, glowStrength: number, selected: boolean,
) {
  if (glowStrength > 0.01) {
    const gR = r * (selected ? 2.5 : 2.0)
    const g = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, gR)
    const gc = selected ? lighten(color, 0.22) : color
    g.addColorStop(0, withAlpha(gc, glowStrength * (selected ? 0.52 : 0.28)))
    g.addColorStop(1, withAlpha(gc, 0))
    ctx.save()
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.arc(cx, cy, gR, 0, TAU)
    ctx.fill()
    ctx.restore()
  }

  const size = r * 2
  const naturalWidth = image.naturalWidth || image.width
  const naturalHeight = image.naturalHeight || image.height
  const cropSide = Math.max(1, Math.min(naturalWidth, naturalHeight))
  const sx = (naturalWidth - cropSide) / 2
  const sy = (naturalHeight - cropSide) / 2

  ctx.save()
  ctx.shadowColor = withAlpha(darken(color, 0.48), selected ? 0.32 : 0.18)
  ctx.shadowBlur = r * (selected ? 0.9 : 0.58)
  ctx.shadowOffsetY = r * 0.18
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, TAU)
  ctx.closePath()
  ctx.clip()
  ctx.drawImage(image, sx, sy, cropSide, cropSide, cx - r, cy - r, size, size)
  ctx.restore()

  const ring = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r)
  ring.addColorStop(0, lighten(color, 0.55))
  ring.addColorStop(1, darken(color, 0.38))
  ctx.save()
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, TAU)
  ctx.lineWidth = Math.max(1.6, r * 0.1)
  ctx.strokeStyle = ring
  ctx.stroke()
  ctx.restore()

  const shine = ctx.createRadialGradient(cx - r * 0.34, cy - r * 0.42, 0, cx - r * 0.08, cy - r * 0.1, r * 0.78)
  shine.addColorStop(0, 'rgba(255,255,255,0.34)')
  shine.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.save()
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, TAU)
  ctx.clip()
  ctx.fillStyle = shine
  ctx.fillRect(cx - r, cy - r, size, size)
  ctx.restore()
}

function projectWorldPoint(point: Point3, camera: ProjectionCamera): ProjectedNode {
  const dx = point.x - camera.x
  const dy = -(point.y - camera.y)
  const dz = point.z - camera.z
  const cosYaw = Math.cos(camera.yaw)
  const sinYaw = Math.sin(camera.yaw)
  const cosPitch = Math.cos(camera.pitch)
  const sinPitch = Math.sin(camera.pitch)

  const yawX = dx * cosYaw - dz * sinYaw
  const yawZ = dx * sinYaw + dz * cosYaw
  const pitchY = dy * cosPitch - yawZ * sinPitch
  const pitchZ = dy * sinPitch + yawZ * cosPitch
  const perspectiveDenominator = CAMERA_FOV + pitchZ
  const scale = CAMERA_FOV / Math.max(CAMERA_FOV * 0.28, perspectiveDenominator)

  return {
    x: camera.width / 2 + yawX * scale * camera.zoom,
    y: camera.height / 2 - pitchY * scale * camera.zoom,
    radius: 0,
    depth: pitchZ,
    scale,
    visible: perspectiveDenominator > CAMERA_FOV * 0.08,
  }
}

function paintFieldBackdrop(
  ctx: CanvasRenderingContext2D,
  camera: ProjectionCamera,
) {
  void ctx
  void camera
}

function projectNode(node: FlowerNode, camera: ProjectionCamera, y: number = node.pos.y): ProjectedNode {
  const projected = projectWorldPoint({ x: node.pos.x, y, z: node.worldZ }, camera)
  return {
    ...projected,
    radius: 22 * node.scale * projected.scale * camera.zoom,
  }
}

function paintNodeShadow(
  ctx: CanvasRenderingContext2D,
  projected: ProjectedNode,
  color: string,
) {
  const normalizedDepth = clamp((projected.depth + 280) / 720, 0, 1)
  const shadowY = projected.y + projected.radius * (0.9 + normalizedDepth * 0.24)
  const shadowRx = projected.radius * (1.16 + normalizedDepth * 0.08)
  const shadowRy = projected.radius * (0.26 + normalizedDepth * 0.04)
  ctx.save()
  ctx.fillStyle = withAlpha(darken(color, 0.4), 0.08 + normalizedDepth * 0.06)
  ctx.shadowColor = withAlpha(darken(color, 0.45), 0.12 + normalizedDepth * 0.04)
  ctx.shadowBlur = 18 + normalizedDepth * 14
  ctx.beginPath()
  ctx.ellipse(projected.x, shadowY, shadowRx, shadowRy, 0, 0, TAU)
  ctx.fill()
  ctx.restore()
}

function paintGlassOverlay(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.save()
  const wash = ctx.createLinearGradient(width * 0.06, 0, width * 0.88, height)
  wash.addColorStop(0, 'rgba(255,255,255,0.20)')
  wash.addColorStop(0.22, 'rgba(255,255,255,0.03)')
  wash.addColorStop(0.62, 'rgba(255,255,255,0)')
  wash.addColorStop(1, 'rgba(255,255,255,0.07)')
  ctx.fillStyle = wash
  ctx.fillRect(0, 0, width, height)
  ctx.restore()
}

function paintNodeLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  zoom: number,
) {
  const fontSize = Math.max(11, 12 / zoom)
  ctx.save()
  ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`
  const width = ctx.measureText(text).width + 20 / zoom
  const height = 24 / zoom
  const radius = 12 / zoom
  const bx = x - width / 2
  const by = y

  ctx.fillStyle = 'rgba(255, 251, 246, 0.92)'
  ctx.strokeStyle = 'rgba(117, 93, 69, 0.18)'
  ctx.lineWidth = 1 / zoom
  ctx.beginPath()
  ctx.moveTo(bx + radius, by)
  ctx.lineTo(bx + width - radius, by)
  ctx.quadraticCurveTo(bx + width, by, bx + width, by + radius)
  ctx.lineTo(bx + width, by + height - radius)
  ctx.quadraticCurveTo(bx + width, by + height, bx + width - radius, by + height)
  ctx.lineTo(bx + radius, by + height)
  ctx.quadraticCurveTo(bx, by + height, bx, by + height - radius)
  ctx.lineTo(bx, by + radius)
  ctx.quadraticCurveTo(bx, by, bx + radius, by)
  ctx.closePath()
  ctx.shadowColor = 'rgba(78, 58, 40, 0.14)'
  ctx.shadowBlur = 18 / zoom
  ctx.shadowOffsetY = 8 / zoom
  ctx.fill()
  ctx.shadowColor = 'transparent'
  ctx.stroke()

  ctx.fillStyle = '#3d2f24'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, x, by + height / 2 + 0.5 / zoom)
  ctx.restore()
}

function _petalPath(ctx: CanvasRenderingContext2D, r: number) {
  const len = r * 0.90, w = r * 0.38
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.bezierCurveTo(-w * 0.90, -len * 0.24, -w * 0.68, -len * 0.76, 0, -len)
  ctx.bezierCurveTo(w * 0.68, -len * 0.76, w * 0.90, -len * 0.24, 0, 0)
  ctx.closePath()
}

function _paintPetal(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, angle: number, r: number,
  base: string, highlight: string, alpha: number,
) {
  const len = r * 0.90, w = r * 0.38, off = r * 0.10
  ctx.save()
  ctx.translate(cx, cy); ctx.rotate(angle + Math.PI / 2); ctx.translate(0, -off)
  const g = ctx.createLinearGradient(0, 0, 0, -len)
  g.addColorStop(0, darken(base, 0.75))
  g.addColorStop(0.18, base)
  g.addColorStop(0.52, highlight)
  g.addColorStop(0.86, base)
  g.addColorStop(1, darken(base, 0.62))
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.bezierCurveTo(-w * 0.90, -len * 0.24, -w * 0.68, -len * 0.76, 0, -len)
  ctx.bezierCurveTo(w * 0.68, -len * 0.76, w * 0.90, -len * 0.24, 0, 0)
  ctx.closePath()
  ctx.globalAlpha = alpha; ctx.fillStyle = g; ctx.fill()
  ctx.globalAlpha = alpha * 0.22
  ctx.strokeStyle = darken(base, 0.58)
  ctx.lineWidth = Math.max(0.3, r * 0.016); ctx.stroke()
  ctx.restore()
}

// ─── Graph / layout math ──────────────────────────────────────────────────────

function getSharedValue(a: string[], b: string[]): string | null {
  const bm = new Map(b.map(v => [normalizeKey(v), v]))
  for (const v of a) { const s = bm.get(normalizeKey(v)); if (s) return s }
  return null
}
function cosineSim(a: VisualizerFlower, b: VisualizerFlower) {
  const ax = a.latent_position.x, ay = a.latent_position.y, az = a.latent_position.z
  const bx = b.latent_position.x, by = b.latent_position.y, bz = b.latent_position.z
  const dot = ax * bx + ay * by + az * bz
  const na = Math.sqrt(ax * ax + ay * ay + az * az), nb = Math.sqrt(bx * bx + by * by + bz * bz)
  return (na === 0 || nb === 0) ? 0 : dot / (na * nb)
}
function richnessScore(f: VisualizerFlower) {
  return f.colors.length * 0.6 + f.meanings.length * 1.25 + f.occasions.length * 0.95 + f.plant_types.length * 0.35 + f.summary.length * 0.5
}
function affinity(a: VisualizerFlower, b: VisualizerFlower) {
  return Math.max(0, cosineSim(a, b)) * 0.56
    + (getSharedValue(a.meanings, b.meanings) ? 0.34 : 0)
    + (getSharedValue(a.occasions, b.occasions) ? 0.22 : 0)
    + (getSharedValue(a.colors, b.colors) ? 0.16 : 0)
    + (normalizeKey(a.primary_meaning) === normalizeKey(b.primary_meaning) ? 0.16 : 0)
    + (normalizeKey(a.primary_occasion) === normalizeKey(b.primary_occasion) ? 0.12 : 0)
    + (normalizeKey(a.primary_color) === normalizeKey(b.primary_color) ? 0.10 : 0)
}
function makeEdgeKey(a: string, b: string) { return [a, b].sort().join('::') }

function buildSemanticGraph(flowers: VisualizerFlower[]): SemanticGraph {
  const candidates: SemanticEdge[] = []
  for (let i = 0; i < flowers.length; i++) for (let j = i + 1; j < flowers.length; j++) {
    const a = flowers[i], b = flowers[j]
    const sim = cosineSim(a, b)
    const sc = getSharedValue(a.colors, b.colors), sm = getSharedValue(a.meanings, b.meanings), so = getSharedValue(a.occasions, b.occasions)
    if (sim < 0.62 && !sc && !sm && !so) continue
    const reason: EdgeReason = sm ? 'meaning' : so ? 'occasion' : sc ? 'color' : 'semantic'
    const color = sm ? '#7f5d49' : so ? '#8a7058' : sc ? '#6f5d4f' : '#5f7862'
    const score = sim * 0.52 + (sm ? 0.32 : 0) + (so ? 0.18 : 0) + (sc ? 0.14 : 0)
    if (score < 0.64) continue
    candidates.push({
      key: makeEdgeKey(a.id, b.id), sourceId: a.id, targetId: b.id, similarity: sim,
      reason, color, width: sim > 0.9 ? 3.4 : sim > 0.8 ? 2.8 : 2.2,
      baseOpacity: clamp(0.34 + sim * 0.24 + (sm ? 0.14 : 0), 0.38, 0.82),
      label: sm ?? so ?? sc ?? null, score,
    })
  }
  const topKeys = new Set<string>()
  flowers.forEach(f => candidates.filter(e => e.sourceId === f.id || e.targetId === f.id)
    .sort((a, b) => b.score - a.score).slice(0, MAX_NEIGHBORS_PER_FLOWER).forEach(e => topKeys.add(e.key)))
  const deg = new Map<string, number>()
  const edges = candidates.filter(e => topKeys.has(e.key)).sort((a, b) => b.score - a.score).filter(e => {
    const sd = deg.get(e.sourceId) ?? 0, td = deg.get(e.targetId) ?? 0
    if (sd >= MAX_NEIGHBORS_PER_FLOWER || td >= MAX_NEIGHBORS_PER_FLOWER) return false
    deg.set(e.sourceId, sd + 1); deg.set(e.targetId, td + 1); return true
  })
  const neighborMap = new Map<string, SemanticEdge[]>()
  flowers.forEach(f => neighborMap.set(f.id, []))
  edges.forEach(e => { neighborMap.get(e.sourceId)?.push(e); neighborMap.get(e.targetId)?.push(e) })
  neighborMap.forEach(arr => arr.sort((a, b) => b.score - a.score))
  return { edges, neighborMap }
}

function importanceMap(flowers: VisualizerFlower[], g: SemanticGraph) {
  const raw = new Map<string, number>(); let max = 0
  flowers.forEach(f => {
    const n = (g.neighborMap.get(f.id) ?? []).slice(0, 4).reduce((t, e) => t + e.score, 0)
    const s = richnessScore(f) + n * 1.4; raw.set(f.id, s); max = Math.max(max, s)
  })
  const out = new Map<string, number>()
  flowers.forEach(f => out.set(f.id, max > 0 ? (raw.get(f.id) ?? 0) / max : 0.5))
  return out
}

function buildBouquets(flowers: VisualizerFlower[], g: SemanticGraph): Bouquet[] {
  if (!flowers.length) return []
  const imp = importanceMap(flowers, g)
  const semanticSpread = new Set(
    flowers.flatMap(f => [
      ...f.meanings.slice(0, 2).map(normalizeKey),
      ...f.occasions.slice(0, 2).map(normalizeKey),
      ...f.colors.slice(0, 2).map(normalizeKey),
    ]).filter(Boolean),
  ).size
  const densityTarget = Math.round(Math.sqrt(flowers.length) * 1.2)
  const semanticTarget = Math.round(semanticSpread / 7)
  const seedTarget = clamp(Math.max(densityTarget, semanticTarget), 8, 14)
  const ranked = [...flowers].sort((a, b) => (imp.get(b.id) ?? 0) - (imp.get(a.id) ?? 0))
  const seeds: VisualizerFlower[] = []
  ranked.forEach(c => {
    if (!seeds.length) { seeds.push(c); return }
    const div = Math.min(...seeds.map(s => affinity(c, s)))
    if (seeds.length < seedTarget && div < 0.72) seeds.push(c)
  })
  while (seeds.length < Math.min(seedTarget, ranked.length)) {
    const fb = ranked.find(f => !seeds.some(s => s.id === f.id)); if (!fb) break; seeds.push(fb)
  }
  const groups = seeds.map(seed => ({ seed, members: [seed] as VisualizerFlower[] }))
  flowers.forEach(f => {
    if (seeds.some(s => s.id === f.id)) return
    let best = groups[0], bestS = -Infinity
    groups.forEach(gr => {
      const s = affinity(f, gr.seed) * 0.68 + gr.members.slice(0, 3).reduce((t, m) => t + affinity(f, m), 0) / Math.max(1, Math.min(gr.members.length, 3)) * 0.32
      if (s > bestS) { bestS = s; best = gr }
    }); best.members.push(f)
  })
  for (let i = groups.length - 1; i >= 0; i--) {
    const gr = groups[i]; if (gr.members.length >= 4 || groups.length <= 7) continue
    let bi = -1, bs = -Infinity
    groups.forEach((c, ci) => { if (ci === i) return; const s = gr.members.reduce((t, m) => t + affinity(m, c.seed), 0) / gr.members.length; if (s > bs) { bs = s; bi = ci } })
    if (bi >= 0) { groups[bi].members.push(...gr.members); groups.splice(i, 1) }
  }
  const minimumBouquets = Math.min(flowers.length, Math.max(8, Math.round(seedTarget * 0.86)))
  while (groups.length < minimumBouquets) {
    let splitIndex = -1
    let splitSeed: VisualizerFlower | null = null
    let splitPriority = -Infinity
    groups.forEach((gr, index) => {
      if (gr.members.length < 8) return
      const candidate = [...gr.members]
        .filter(member => member.id !== gr.seed.id)
        .sort((a, b) => affinity(a, gr.seed) - affinity(b, gr.seed))[0]
      if (!candidate) return
      const variance = gr.members.reduce((total, member) => total + (1 - affinity(member, gr.seed)), 0) / gr.members.length
      const priority = gr.members.length * 0.7 + variance * 10
      if (priority > splitPriority) {
        splitPriority = priority
        splitIndex = index
        splitSeed = candidate
      }
    })
    if (splitIndex < 0 || !splitSeed) break
    const altSeed = splitSeed as VisualizerFlower
    const source = groups[splitIndex]
    const primaryMembers: VisualizerFlower[] = [source.seed]
    const secondaryMembers: VisualizerFlower[] = [altSeed]
    source.members.forEach(member => {
      if (member.id === source.seed.id || member.id === altSeed.id) return
      const primaryScore = affinity(member, source.seed)
      const secondaryScore = affinity(member, altSeed)
      if (secondaryScore > primaryScore) secondaryMembers.push(member)
      else primaryMembers.push(member)
    })
    if (primaryMembers.length < 3 || secondaryMembers.length < 3) break
    groups[splitIndex] = { seed: source.seed, members: primaryMembers }
    groups.push({ seed: altSeed, members: secondaryMembers })
  }
  const dom = (ms: VisualizerFlower[], pick: (f: VisualizerFlower) => string[], fb: string) => {
    const counts = new Map<string, { raw: string; count: number }>()
    ms.forEach(m => pick(m).slice(0, 2).forEach(v => { const k = normalizeKey(v); if (!k) return; const c = counts.get(k); counts.set(k, { raw: v, count: (c?.count ?? 0) + 1 }) }))
    return [...counts.values()].sort((a, b) => b.count - a.count)[0]?.raw ?? fb
  }
  return groups.map(({ seed, members }) => {
    const sorted = [...members].sort((a, b) => (imp.get(b.id) ?? 0) - (imp.get(a.id) ?? 0))
    const meaning = dom(sorted, f => f.meanings, seed.primary_meaning)
    const occasion = dom(sorted, f => f.occasions, seed.primary_occasion)
    const primaryColor = dom(sorted, f => f.colors, seed.primary_color)
    const labelBasis = normalizeKey(meaning) !== 'general' ? meaning : normalizeKey(occasion) !== 'everyday' ? occasion : primaryColor
    const descriptor = [
      normalizeKey(primaryColor) !== 'neutral' ? `${toTitleCase(primaryColor)} tones` : null,
      normalizeKey(occasion) !== 'everyday' ? `for ${occasion}` : null,
      `${sorted.length} blooms`,
    ].filter(Boolean).join(' · ')
    return {
      id: `bouquet-${seed.id}`, label: `${toTitleCase(labelBasis)} Bouquet`, descriptor,
      color: getColorHex(primaryColor), meaning, occasion, primaryColor,
      centroidFlowerId: sorted[0]?.id ?? seed.id,
      importance: sorted.reduce((t, m) => t + (imp.get(m.id) ?? 0), 0) / sorted.length,
      memberIds: sorted.map(m => m.id), previewIds: sorted.slice(0, 5).map(m => m.id),
    }
  }).sort((a, b) => b.importance - a.importance)
}

function getBouquetByFlowerId(bouquets: Bouquet[]) {
  const m = new Map<string, string>()
  bouquets.forEach(b => b.memberIds.forEach(id => m.set(id, b.id))); return m
}
// ─── 2D Layout ────────────────────────────────────────────────────────────────

interface Layout2D {
  anchorPos: Map<string, Point2>
  flowerPos: Map<string, Point2>
  visibleEdgeKeys: Set<string>
  viewCenter: Point2
  viewZoom: number
}

function computeLayout(
  flowers: VisualizerFlower[], bouquets: Bouquet[], g: SemanticGraph,
  focusFlowerId: string | null, focusBouquetId: string | null,
  W: number, H: number,
): Layout2D {
  void bouquets
  void focusBouquetId

  const spreadX = Math.max(560, W * 0.86)
  const spreadY = Math.max(420, H * 0.74)
  const anchorPos = new Map<string, Point2>()
  const flowerPos = new Map<string, Point2>()
  const states = flowers.map(flower => {
    const seed = hashString(flower.id)
    const jitterX = (((seed & 1023) / 1023) - 0.5) * 46
    const jitterY = ((((seed >>> 10) & 1023) / 1023) - 0.5) * 38
    return {
      flower,
      pos: {
        x: flower.latent_position.x * spreadX + flower.latent_position.z * spreadX * 0.14 + jitterX,
        y: -flower.latent_position.z * spreadY + flower.latent_position.x * spreadY * 0.08 + jitterY,
      },
      vel: { x: 0, y: 0 },
    }
  })

  const stateById = new Map(states.map(state => [state.flower.id, state]))
  const stateIndexById = new Map(states.map((state, index) => [state.flower.id, index]))
  for (let step = 0; step < 92; step++) {
    const forces = states.map(() => ({ x: 0, y: 0 }))

    for (let a = 0; a < states.length; a++) for (let b = a + 1; b < states.length; b++) {
      const dx = states[b].pos.x - states[a].pos.x
      const dy = states[b].pos.y - states[a].pos.y
      const d2 = dx * dx + dy * dy + 0.01
      const d = Math.sqrt(d2)
      const rep = 26000 / d2
      const fx = dx / d * rep
      const fy = dy / d * rep
      forces[a].x -= fx
      forces[a].y -= fy
      forces[b].x += fx
      forces[b].y += fy
    }

    g.edges.forEach(edge => {
      const src = stateById.get(edge.sourceId)
      const tgt = stateById.get(edge.targetId)
      if (!src || !tgt) return
      const si = stateIndexById.get(edge.sourceId)
      const ti = stateIndexById.get(edge.targetId)
      if (si === undefined || ti === undefined) return
      const dx = tgt.pos.x - src.pos.x
      const dy = tgt.pos.y - src.pos.y
      const d = Math.max(0.1, Math.sqrt(dx * dx + dy * dy))
      const preferred = 110 + (1 - edge.similarity) * 120
      const pull = (d - preferred) * (0.010 + edge.similarity * 0.024)
      forces[si].x += dx / d * pull
      forces[si].y += dy / d * pull
      forces[ti].x -= dx / d * pull
      forces[ti].y -= dy / d * pull
    })

    states.forEach((state, index) => {
      const homeX = state.flower.latent_position.x * spreadX + state.flower.latent_position.z * spreadX * 0.14
      const homeY = -state.flower.latent_position.z * spreadY + state.flower.latent_position.x * spreadY * 0.08
      forces[index].x += (homeX - state.pos.x) * 0.016
      forces[index].y += (homeY - state.pos.y) * 0.016
      state.vel.x = (state.vel.x + forces[index].x) * 0.82
      state.vel.y = (state.vel.y + forces[index].y) * 0.82
      state.pos.x += state.vel.x
      state.pos.y += state.vel.y
    })
  }

  states.forEach(state => flowerPos.set(state.flower.id, state.pos))

  const visibleEdgeKeys = new Set<string>()
  if (focusFlowerId) {
    ;(g.neighborMap.get(focusFlowerId) ?? []).slice(0, 8).forEach(edge => visibleEdgeKeys.add(edge.key))
  } else {
    g.edges.forEach(edge => visibleEdgeKeys.add(edge.key))
  }

  const points = [...flowerPos.values()]
  let minX = -spreadX
  let maxX = spreadX
  let minY = -spreadY
  let maxY = spreadY
  if (points.length) {
    minX = Math.min(...points.map(point => point.x))
    maxX = Math.max(...points.map(point => point.x))
    minY = Math.min(...points.map(point => point.y))
    maxY = Math.max(...points.map(point => point.y))
  }
  const boundsCenter = { x: (minX + maxX) * 0.5, y: (minY + maxY) * 0.5 }
  const focusPt = focusFlowerId ? flowerPos.get(focusFlowerId) ?? null : null
  const viewCenter = focusPt
    ? { x: lerp(boundsCenter.x, focusPt.x, 0.68), y: lerp(boundsCenter.y, focusPt.y, 0.68) }
    : boundsCenter
  const padX = 240
  const padY = 220
  const fitZoom = Math.min(
    W / Math.max(1, maxX - minX + padX),
    H / Math.max(1, maxY - minY + padY),
  )
  const viewZoom = clamp(
    focusFlowerId ? fitZoom * 1.16 : fitZoom * 0.88,
    0.20,
    focusFlowerId ? 1.28 : 0.84,
  )
  return { anchorPos, flowerPos, visibleEdgeKeys, viewCenter, viewZoom }
}

function getViewportNodes(
  nodes: FlowerNode[],
  semanticGraph: SemanticGraph,
  selectedFlowerId: string | null,
): FlowerNode[] {
  if (!selectedFlowerId) return nodes

  const selectedNode = nodes.find(candidate => candidate.data.id === selectedFlowerId)
  if (!selectedNode) return nodes

  const nodeById = new Map(nodes.map(node => [node.data.id, node]))
  const relatedNodes = (semanticGraph.neighborMap.get(selectedFlowerId) ?? [])
    .slice(0, 4)
    .map(edge => {
      const relatedId = edge.sourceId === selectedFlowerId ? edge.targetId : edge.sourceId
      return nodeById.get(relatedId) ?? null
    })
    .filter(Boolean) as FlowerNode[]

  return [selectedNode, ...relatedNodes]
}

function fitProjectedViewport(
  nodes: FlowerNode[],
  semanticGraph: SemanticGraph,
  selectedFlowerId: string | null,
  width: number,
  height: number,
) {
  const viewportNodes = getViewportNodes(nodes, semanticGraph, selectedFlowerId)
  if (!viewportNodes.length) {
    return {
      camX: 0,
      camY: 0,
      camZ: 0,
      zoom: 0.74,
      yaw: DEFAULT_ORBIT_YAW,
      pitch: DEFAULT_ORBIT_PITCH,
    }
  }

  const camZ = 0
  const yaw = DEFAULT_ORBIT_YAW
  const pitch = DEFAULT_ORBIT_PITCH
  const avgX = viewportNodes.reduce((total, node) => total + node.targetPos.x, 0) / viewportNodes.length
  const avgY = viewportNodes.reduce((total, node) => total + node.targetPos.y, 0) / viewportNodes.length
  const avgZ = viewportNodes.reduce((total, node) => total + node.worldZ, 0) / viewportNodes.length

  const camX = avgX - Math.tan(yaw) * (avgZ - camZ)
  const avgYawZ = (avgX - camX) * Math.sin(yaw) + (avgZ - camZ) * Math.cos(yaw)
  const camY = avgY + Math.tan(pitch) * avgYawZ
  const projectionCamera = {
    x: camX,
    y: camY,
    z: camZ,
    zoom: 1,
    yaw,
    pitch,
    width,
    height,
  }
  const projectedNodes = viewportNodes
    .map(node => projectNode(node, projectionCamera))
    .filter(projected => projected.visible)

  if (!projectedNodes.length) {
    return {
      camX,
      camY,
      camZ,
      zoom: 0.74,
      yaw,
      pitch,
    }
  }

  const minX = Math.min(...projectedNodes.map(node => node.x - node.radius))
  const maxX = Math.max(...projectedNodes.map(node => node.x + node.radius))
  const minY = Math.min(...projectedNodes.map(node => node.y - node.radius))
  const maxY = Math.max(...projectedNodes.map(node => node.y + node.radius))
  const padX = selectedFlowerId ? 200 : 240
  const padY = selectedFlowerId ? 190 : 220
  const usableWidth = Math.max(1, width - padX)
  const usableHeight = Math.max(1, height - padY)
  const fitZoom = Math.min(
    usableWidth / Math.max(1, maxX - minX),
    usableHeight / Math.max(1, maxY - minY),
  )
  const zoom = clamp(
    fitZoom * (selectedFlowerId ? 1.02 : 0.94),
    0.35,
    selectedFlowerId ? 2.2 : 1.24,
  )

  return {
    camX,
    camY,
    camZ,
    zoom,
    yaw,
    pitch,
  }
}

// ─── Canvas Visualizer component ──────────────────────────────────────────────

interface VisualizerCanvasProps {
  isActive: boolean
  flowers: VisualizerFlower[]; semanticGraph: SemanticGraph; bouquets: Bouquet[]
  selectedFlowerId: string | null; selectedBouquetId: string | null
  viewportResetToken: number
  onFlowerSelect(id: string): void; onBouquetSelect(id: string): void
  onSceneStatusChange(s: SceneStatus, msg: string): void
}

const VisualizerCanvas = memo(function VisualizerCanvas(props: VisualizerCanvasProps) {
  const { isActive, flowers, semanticGraph, bouquets, selectedFlowerId, selectedBouquetId, viewportResetToken, onFlowerSelect, onBouquetSelect, onSceneStatusChange } = props
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const sr = useRef({
    nodes: [] as FlowerNode[], anchors: [] as BouquetAnchor[], links: [] as EdgeLink[],
    images: new Map<string, HTMLImageElement>(),
    camX: 0, camY: 0, camZ: 0, targetCamX: 0, targetCamY: 0, targetCamZ: 0,
    zoom: 0.74, targetZoom: 0.74,
    orbitYaw: DEFAULT_ORBIT_YAW, targetOrbitYaw: DEFAULT_ORBIT_YAW,
    orbitPitch: DEFAULT_ORBIT_PITCH, targetOrbitPitch: DEFAULT_ORBIT_PITCH,
    isPanning: false, panStartX: 0, panStartY: 0,
    orbitStartYaw: DEFAULT_ORBIT_YAW, orbitStartPitch: DEFAULT_ORBIT_PITCH,
    hoveredFlowerId: null as string | null, hoveredBouquetId: null as string | null,
    selectedFlowerId: null as string | null, selectedBouquetId: null as string | null,
    W: 800, H: 600, dpr: 1, rafId: 0, t: 0, prefersReducedMotion: false,
  })

  const recompute = useCallback(() => {
    const s = sr.current
    const activeFlowerId = s.hoveredFlowerId ?? s.selectedFlowerId
    const layout = computeLayout(flowers, bouquets, semanticGraph, s.selectedFlowerId, s.selectedBouquetId, s.W, s.H)
    const neighborIds = new Set(activeFlowerId ? (semanticGraph.neighborMap.get(activeFlowerId) ?? []).slice(0, 6).map(e => e.sourceId === activeFlowerId ? e.targetId : e.sourceId) : [])
    s.nodes.forEach(n => {
      const pos = layout.flowerPos.get(n.data.id); if (pos) n.targetPos = { ...pos }
      const isSel = n.data.id === s.selectedFlowerId, isFoc = n.data.id === activeFlowerId
      const isNbr = neighborIds.has(n.data.id)
      n.targetScale = isSel ? 1.76 : isFoc ? 1.48 : isNbr ? 1.18 : 0.88 + n.importance * 0.16
      n.targetGlow = isSel ? 1.0 : isFoc ? 0.78 : isNbr ? 0.42 : 0.08
    })
    s.links.forEach(l => {
      const touchesFocus = activeFlowerId !== null && (l.data.sourceId === activeFlowerId || l.data.targetId === activeFlowerId)
      const touchesNeighbor = activeFlowerId !== null && (neighborIds.has(l.data.sourceId) || neighborIds.has(l.data.targetId))
      l.targetOpacity = activeFlowerId
        ? touchesFocus
          ? Math.min(0.92, l.data.baseOpacity + 0.16)
          : touchesNeighbor
            ? Math.min(0.34, l.data.baseOpacity * 0.48)
            : Math.min(0.12, l.data.baseOpacity * 0.18)
        : layout.visibleEdgeKeys.has(l.data.key)
          ? Math.min(0.22, l.data.baseOpacity * 0.34)
          : 0
    })
  }, [flowers, bouquets, semanticGraph])

  const centerViewport = useCallback(() => {
    const s = sr.current
    const fittedViewport = fitProjectedViewport(
      s.nodes,
      semanticGraph,
      s.selectedFlowerId,
      s.W,
      s.H,
    )
    s.targetCamX = fittedViewport.camX
    s.targetCamY = fittedViewport.camY
    s.targetCamZ = fittedViewport.camZ
    s.targetZoom = clamp(fittedViewport.zoom, 0.35, 2.8)
    s.targetOrbitYaw = fittedViewport.yaw
    s.targetOrbitPitch = fittedViewport.pitch
  }, [flowers, bouquets, semanticGraph])

  const setOverviewViewport = useCallback(() => {
    const s = sr.current
    const fittedViewport = fitProjectedViewport(
      s.nodes,
      semanticGraph,
      null,
      s.W,
      s.H,
    )
    s.targetCamX = fittedViewport.camX
    s.targetCamY = fittedViewport.camY
    s.targetCamZ = fittedViewport.camZ
    s.targetZoom = clamp(fittedViewport.zoom, 0.35, 2.8)
    s.targetOrbitYaw = fittedViewport.yaw
    s.targetOrbitPitch = fittedViewport.pitch
  }, [flowers, bouquets, semanticGraph])

  const focusNode = useCallback((node: FlowerNode) => {
    const s = sr.current
    const yaw = s.targetOrbitYaw
    const pitch = s.targetOrbitPitch
    const camZ = 0
    const camX = node.targetPos.x - Math.tan(yaw) * (node.worldZ - camZ)
    const yawZ = (node.targetPos.x - camX) * Math.sin(yaw) + (node.worldZ - camZ) * Math.cos(yaw)
    const camY = node.targetPos.y + Math.tan(pitch) * yawZ
    s.targetCamX = camX
    s.targetCamY = camY
    s.targetCamZ = camZ
  }, [])

  const focusSelectedFlower = useCallback(() => {
    const s = sr.current
    if (!s.selectedFlowerId) return false
    const selectedNode = s.nodes.find(candidate => candidate.data.id === s.selectedFlowerId)
    if (!selectedNode) return false
    focusNode(selectedNode)
    return true
  }, [focusNode])

  const focusSelectedNeighborhood = useCallback(() => {
    const s = sr.current
    if (!s.selectedFlowerId) return false
    const selectedNode = s.nodes.find(candidate => candidate.data.id === s.selectedFlowerId)
    if (!selectedNode) return false

    const relatedEdges = (semanticGraph.neighborMap.get(s.selectedFlowerId) ?? []).slice(0, 4)
    if (relatedEdges.length === 0) return false

    const relatedNodes = relatedEdges
      .map(edge => {
        const relatedId = edge.sourceId === s.selectedFlowerId ? edge.targetId : edge.sourceId
        return s.nodes.find(candidate => candidate.data.id === relatedId) ?? null
      })
      .filter(Boolean) as FlowerNode[]

    const cluster = [selectedNode, ...relatedNodes]
    s.targetCamX = cluster.reduce((total, node) => total + node.targetPos.x, 0) / cluster.length
    s.targetCamY = cluster.reduce((total, node) => total + node.targetPos.y, 0) / cluster.length
    s.targetCamZ = cluster.reduce((total, node) => total + node.worldZ, 0) / cluster.length
    return true
  }, [semanticGraph])

  useEffect(() => {
    sr.current.selectedFlowerId = selectedFlowerId
    sr.current.selectedBouquetId = selectedBouquetId
    recompute()
  }, [selectedFlowerId, selectedBouquetId, recompute])

  useEffect(() => {
    const s = sr.current
    s.selectedFlowerId = selectedFlowerId
    s.selectedBouquetId = selectedBouquetId
    if (selectedFlowerId) {
      centerViewport()
      focusSelectedFlower()
      return
    }
    setOverviewViewport()
  }, [centerViewport, focusSelectedFlower, selectedBouquetId, selectedFlowerId, setOverviewViewport, viewportResetToken])

  useEffect(() => {
    if (!selectedFlowerId) return
    const s = sr.current
    s.targetZoom = clamp(Math.max(s.zoom * 1.18, 1.18), 0.35, 2.2)
  }, [selectedFlowerId])

  useEffect(() => {
    if (!isActive) return

    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const canvasEl = canvas
    const s = sr.current
    s.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    s.dpr = window.devicePixelRatio || 1
    const imp = importanceMap(flowers, semanticGraph)

    s.nodes = flowers.map((f, i) => {
      const seed = hashString(f.id)
      return {
        data: f, pos: { x: 0, y: 0 }, targetPos: { x: 0, y: 0 },
        scale: 1, targetScale: 1, glow: 0, targetGlow: 0,
        petalCount: 5 + (seed % 4),
        petalAngleOffset: ((seed >>> 3) % 360) / 360 * TAU,
        spinAngle: ((seed >>> 11) % 360) / 360 * TAU,
        driftPhase: i * 0.67 + ((seed >>> 19) % 100) / 100,
        importance: imp.get(f.id) ?? 0.5,
        bouquetId: null,
        worldZ: f.latent_position.y * NODE_DEPTH_SPREAD,
      }
    })
    s.anchors = []
    s.links = semanticGraph.edges.map(e => ({ data: e, opacity: 0, targetOpacity: 0 }))
    s.images = new Map(
      flowers
        .filter(flower => Boolean(flower.image_url))
        .map(flower => {
          const image = new Image()
          image.decoding = 'async'
          image.src = flower.image_url!
          return [flower.id, image] as const
        }),
    )

    const resize = () => {
      s.W = container.clientWidth || 800; s.H = container.clientHeight || 600; s.dpr = window.devicePixelRatio || 1
      canvasEl.width = s.W * s.dpr; canvasEl.height = s.H * s.dpr
      canvasEl.style.width = s.W + 'px'; canvasEl.style.height = s.H + 'px'
    }
    resize(); recompute(); setOverviewViewport()
    s.nodes.forEach(n => { n.pos = { ...n.targetPos } })
    s.camX = s.targetCamX; s.camY = s.targetCamY; s.camZ = s.targetCamZ; s.zoom = s.targetZoom
    s.orbitYaw = s.targetOrbitYaw; s.orbitPitch = s.targetOrbitPitch

    function hitTest(sx: number, sy: number) {
      const camera = {
        x: s.camX, y: s.camY, z: s.camZ,
        zoom: s.zoom, yaw: s.orbitYaw, pitch: s.orbitPitch,
        width: s.W, height: s.H,
      }
      const sorted = [...s.nodes]
        .map(node => ({ node, projected: projectNode(node, camera) }))
        .filter(entry => entry.projected.visible)
        .sort((a, b) => b.projected.scale - a.projected.scale)
      for (const { node, projected } of sorted) {
        if ((sx - projected.x) ** 2 + (sy - projected.y) ** 2 < projected.radius * projected.radius) return { flowerId: node.data.id, bouquetId: null }
      }
      return { flowerId: null, bouquetId: null }
    }

    let didPan = false
    let restoredOverviewOnDrag = false
    const onDown = (e: PointerEvent) => {
      s.isPanning = true; s.panStartX = e.clientX; s.panStartY = e.clientY
      s.orbitStartYaw = s.targetOrbitYaw
      s.orbitStartPitch = s.targetOrbitPitch
      didPan = false
      restoredOverviewOnDrag = false
      canvasEl.setPointerCapture(e.pointerId)
    }
    const onMove = (e: PointerEvent) => {
      const rect = canvasEl.getBoundingClientRect()
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top
      if (s.isPanning) {
        const dx = e.clientX - s.panStartX, dy = e.clientY - s.panStartY
        if (Math.abs(dx) + Math.abs(dy) > 4) {
          didPan = true
          if (s.selectedFlowerId && !restoredOverviewOnDrag) {
            setOverviewViewport()
            restoredOverviewOnDrag = true
          }
        }
        const nextYaw = s.orbitStartYaw + dx * 0.0052
        const nextPitch = clamp(s.orbitStartPitch + dy * 0.0038, MIN_ORBIT_PITCH, MAX_ORBIT_PITCH)
        s.orbitYaw = nextYaw; s.targetOrbitYaw = nextYaw
        s.orbitPitch = nextPitch; s.targetOrbitPitch = nextPitch
        canvasEl.style.cursor = 'grabbing'
      } else {
        const hit = hitTest(sx, sy)
        canvasEl.style.cursor = hit.flowerId ? 'pointer' : 'grab'
        if (hit.flowerId !== s.hoveredFlowerId || hit.bouquetId !== s.hoveredBouquetId) {
          s.hoveredFlowerId = hit.flowerId; s.hoveredBouquetId = hit.bouquetId; recompute()
        }
      }
    }
    const onUp = (e: PointerEvent) => {
      s.isPanning = false; canvasEl.style.cursor = 'grab'
      if (!didPan) {
        const rect = canvasEl.getBoundingClientRect()
        const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top)
        if (hit.flowerId) onFlowerSelect(hit.flowerId)
      }
    }
    const onLeave = () => { s.isPanning = false; s.hoveredFlowerId = null; s.hoveredBouquetId = null; canvasEl.style.cursor = 'grab'; recompute() }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const factor = e.deltaY > 0 ? 0.92 : 1.09
      const nz = clamp(s.zoom * factor, 0.28, 3.8)
      if (!focusSelectedFlower()) focusSelectedNeighborhood()
      s.zoom = nz; s.targetZoom = nz
    }
    canvasEl.addEventListener('pointerdown', onDown)
    canvasEl.addEventListener('pointermove', onMove)
    canvasEl.addEventListener('pointerup', onUp)
    canvasEl.addEventListener('pointerleave', onLeave)
    canvasEl.addEventListener('wheel', onWheel, { passive: false })

    function frame(ts: number) {
      s.t = ts * 0.001
      const ctx = canvasEl.getContext('2d'); if (!ctx) { s.rafId = requestAnimationFrame(frame); return }
      s.camX = lerp(s.camX, s.targetCamX, 0.09); s.camY = lerp(s.camY, s.targetCamY, 0.09); s.camZ = lerp(s.camZ, s.targetCamZ, 0.09)
      s.orbitYaw = lerp(s.orbitYaw, s.targetOrbitYaw, 0.12)
      s.orbitPitch = lerp(s.orbitPitch, s.targetOrbitPitch, 0.12)
      s.zoom = lerp(s.zoom, s.targetZoom, 0.09)
      s.nodes.forEach(n => {
        n.pos.x = lerp(n.pos.x, n.targetPos.x, 0.11); n.pos.y = lerp(n.pos.y, n.targetPos.y, 0.11)
        n.scale = lerp(n.scale, n.targetScale, 0.13); n.glow = lerp(n.glow, n.targetGlow, 0.15)
        if (!s.prefersReducedMotion) n.spinAngle += 0.0020 + n.driftPhase * 0.00012
      })
      s.links.forEach(l => { l.opacity = lerp(l.opacity, l.targetOpacity, 0.13) })

      ctx.save()
      ctx.scale(s.dpr, s.dpr)
      ctx.clearRect(0, 0, s.W, s.H)
      const screenGlow = ctx.createRadialGradient(s.W * 0.22, s.H * 0.16, 0, s.W * 0.22, s.H * 0.16, Math.max(s.W, s.H) * 0.62)
      screenGlow.addColorStop(0, 'rgba(255,255,255,0.24)')
      screenGlow.addColorStop(1, 'rgba(255,255,255,0)')
      ctx.fillStyle = screenGlow
      ctx.fillRect(0, 0, s.W, s.H)

      const camera = {
        x: s.camX, y: s.camY, z: s.camZ,
        zoom: s.zoom, yaw: s.orbitYaw, pitch: s.orbitPitch,
        width: s.W, height: s.H,
      }
      paintFieldBackdrop(ctx, camera)

      const projectedNodes = s.nodes
        .map(node => {
          const drift = s.prefersReducedMotion ? 0 : Math.sin(s.t * 0.55 + node.driftPhase) * 1.6
          return {
            node,
            projected: projectNode(node, camera, node.pos.y + drift),
            drift,
          }
        })
        .filter(entry => entry.projected.visible)
      const projectedNodeById = new Map(projectedNodes.map(entry => [entry.node.data.id, entry]))

      const nodeMap = new Map(s.nodes.map(n => [n.data.id, n]))
      s.links.forEach(l => {
        if (l.opacity < 0.01) return
        const src = nodeMap.get(l.data.sourceId), tgt = nodeMap.get(l.data.targetId)
        if (!src || !tgt) return
        const projectedSrc = projectedNodeById.get(src.data.id)?.projected ?? projectNode(src, camera)
        const projectedTgt = projectedNodeById.get(tgt.data.id)?.projected ?? projectNode(tgt, camera)
        const avgDepth = clamp((projectedSrc.scale + projectedTgt.scale) * 0.5, 0.25, 1.8)
        if (!projectedSrc.visible && !projectedTgt.visible) return
        const srcColor = getFlowerColorHex(src.data)
        const tgtColor = getFlowerColorHex(tgt.data)
        const gradient = ctx.createLinearGradient(projectedSrc.x, projectedSrc.y, projectedTgt.x, projectedTgt.y)
        gradient.addColorStop(0, withAlpha(srcColor, l.opacity * 0.85))
        gradient.addColorStop(0.5, withAlpha(lighten(l.data.color, 0.12), l.opacity))
        gradient.addColorStop(1, withAlpha(tgtColor, l.opacity * 0.85))
        const mx = (projectedSrc.x + projectedTgt.x) / 2, my = (projectedSrc.y + projectedTgt.y) / 2
        const ox = (projectedTgt.y - projectedSrc.y) * 0.12
        const oy = (projectedSrc.x - projectedTgt.x) * 0.12
        ctx.save()
        ctx.beginPath()
        ctx.moveTo(projectedSrc.x, projectedSrc.y)
        ctx.bezierCurveTo(mx + ox, my + oy, mx + ox, my + oy, projectedTgt.x, projectedTgt.y)
        ctx.strokeStyle = withAlpha(l.data.color, l.opacity * 0.2)
        ctx.lineWidth = l.data.width * (1.2 + avgDepth * 1.15)
        ctx.lineCap = 'round'
        ctx.shadowColor = withAlpha(l.data.color, l.opacity * 0.16)
        ctx.shadowBlur = 8 + avgDepth * 9
        ctx.stroke()
        ctx.shadowColor = 'transparent'
        ctx.strokeStyle = gradient
        ctx.lineWidth = l.data.width * (0.7 + avgDepth * 0.42)
        ctx.stroke()
        ctx.strokeStyle = `rgba(255,255,255,${l.opacity * 0.2})`
        ctx.lineWidth = Math.max(0.45, l.data.width * (0.16 + avgDepth * 0.08))
        ctx.stroke()
        ctx.restore()
      })

      const sorted = projectedNodes
        .sort((a, b) => a.projected.scale - b.projected.scale)
      sorted.forEach(({ node: n, projected }) => {
        const image = s.images.get(n.data.id)
        paintNodeShadow(ctx, projected, getFlowerColorHex(n.data))
        if (image && image.complete && image.naturalWidth > 0) {
          paintFlowerImage(ctx, image, projected.x, projected.y, projected.radius, getFlowerColorHex(n.data), n.glow, n.data.id === s.selectedFlowerId)
          return
        }
        paintFlower(ctx, projected.x, projected.y, projected.radius, getFlowerColorHex(n.data),
          n.petalCount, n.spinAngle + n.petalAngleOffset, n.glow, n.data.id === s.selectedFlowerId)
      })

      const focusNode = s.nodes.find(n => n.data.id === (s.hoveredFlowerId ?? s.selectedFlowerId))
      if (focusNode) {
        const projected = projectNode(focusNode, camera)
        const labelY = projected.y + projected.radius * 1.22
        paintNodeLabel(ctx, focusNode.data.name, projected.x, labelY, s.zoom)
      }

      paintGlassOverlay(ctx, s.W, s.H)
      ctx.restore()
      s.rafId = requestAnimationFrame(frame)
    }
    s.rafId = requestAnimationFrame(frame)

    const ro = new ResizeObserver(() => {
      resize()
      recompute()
      if (s.isPanning) return
      if (s.selectedFlowerId) centerViewport()
      else setOverviewViewport()
    })
    ro.observe(container)
    onSceneStatusChange('ready', 'Drag to rotate the field · scroll to zoom · click a flower to move closer')

    return () => {
      cancelAnimationFrame(s.rafId); ro.disconnect()
      canvasEl.removeEventListener('pointerdown', onDown)
      canvasEl.removeEventListener('pointermove', onMove)
      canvasEl.removeEventListener('pointerup', onUp)
      canvasEl.removeEventListener('pointerleave', onLeave)
      canvasEl.removeEventListener('wheel', onWheel)
    }
  }, [isActive, bouquets, flowers, semanticGraph, recompute, centerViewport, focusSelectedFlower, focusSelectedNeighborhood, onFlowerSelect, onBouquetSelect, onSceneStatusChange, setOverviewViewport])

  return (
    <div ref={containerRef} className="viz-canvas__mount" style={{ position: 'absolute', inset: 0 }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%', cursor: 'grab' }} />
    </div>
  )
})

function Visualizer3D({ isActive = true }: Visualizer3DProps): JSX.Element {
  const [flowers, setFlowers] = useState<VisualizerFlower[]>([])
  const [selectedFlowerId, setSelectedFlowerId] = useState<string | null>(null)
  const [selectedBouquetId, setSelectedBouquetId] = useState<string | null>(null)
  const [viewportResetToken, setViewportResetToken] = useState(0)
  const [sceneStatus, setSceneStatus] = useState<SceneStatus>('loading')
  const [statusMessage, setStatusMessage] = useState('Loading 3D Visualizer...')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [bouquetInsights, setBouquetInsights] = useState<BouquetInsightsResponse | null>(null)
  const [bouquetInsightsStatus, setBouquetInsightsStatus] = useState<InsightsStatus>('idle')
  const [bouquetInsightsError, setBouquetInsightsError] = useState<string | null>(null)
  const shellRef = useRef<HTMLElement | null>(null)
  const healthListRef = useRef<HTMLDivElement | null>(null)
  const recommendationListRef = useRef<HTMLDivElement | null>(null)
  const introAnimatedRef = useRef(false)
  const bouquetInsightsCacheRef = useRef(new Map<string, BouquetInsightsResponse>())

  useEffect(() => {
    if (!isActive || flowers.length > 0) return

    let cancelled = false
    fetch('/api/visualizer-flowers?limit=96')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((p: VisualizerFlowersResponse) => { if (!cancelled) setFlowers(p.flowers) })
      .catch(err => {
        if (cancelled) return
        setLoadError(err instanceof Error ? `Could not load bouquet data: ${err.message}` : 'Could not load bouquet data.')
        setSceneStatus('error'); setStatusMessage('Semantic bouquet data failed to load.')
      })
    return () => { cancelled = true }
  }, [flowers.length, isActive])

  const semanticGraph = useMemo(() => buildSemanticGraph(flowers), [flowers])
  const bouquets = useMemo(() => buildBouquets(flowers, semanticGraph), [flowers, semanticGraph])
  const bouquetByFlowerId = useMemo(() => getBouquetByFlowerId(bouquets), [bouquets])
  const flowersById = useMemo(() => new Map(flowers.map(f => [f.id, f])), [flowers])
  const flowersByScientificName = useMemo(() => {
    const byScientificName = new Map<string, VisualizerFlower>()
    flowers.forEach(flower => {
      const scientificName = normalizeKey(flower.scientific_name)
      if (scientificName) byScientificName.set(scientificName, flower)
    })
    return byScientificName
  }, [flowers])

  useEffect(() => { setSelectedBouquetId(c => c && bouquets.some(b => b.id === c) ? c : null) }, [bouquets])
  useEffect(() => { if (!selectedFlowerId) return; const bid = bouquetByFlowerId.get(selectedFlowerId); if (bid) setSelectedBouquetId(bid) }, [bouquetByFlowerId, selectedFlowerId])

  const handleSceneStatus = useCallback((s: SceneStatus, m: string) => { setSceneStatus(s); setStatusMessage(m) }, [])
  const handleFlowerSelect = useCallback((id: string) => { setSelectedFlowerId(id); setSelectedBouquetId(bouquetByFlowerId.get(id) ?? null) }, [bouquetByFlowerId])
  const handleBouquetSelect = useCallback((id: string) => { setSelectedBouquetId(id); setSelectedFlowerId(null) }, [])
  const handleResetFocus = useCallback(() => {
    setSelectedFlowerId(null)
    setSelectedBouquetId(null)
    setViewportResetToken(token => token + 1)
  }, [])

  const selectedFlower = useMemo(() => flowers.find(f => f.id === selectedFlowerId) ?? null, [flowers, selectedFlowerId])
  const selectedBouquet = useMemo(() => bouquets.find(b => b.id === selectedBouquetId) ?? null, [bouquets, selectedBouquetId])
  const showSidebar = Boolean(selectedFlower)
  const bouquetMembers = useMemo(() => {
    if (!selectedBouquet) return []
    return selectedBouquet.memberIds.map(id => flowersById.get(id)).filter(Boolean) as VisualizerFlower[]
  }, [flowersById, selectedBouquet])
  const bouquetScientificNames = useMemo(() => {
    return [...new Set(
      bouquetMembers
        .map(flower => flower.scientific_name.trim())
        .filter(Boolean),
    )]
  }, [bouquetMembers])
  const atlasRecommendations = useMemo(() => {
    return (bouquetInsights?.recommendations ?? []).map(recommendation => ({
      recommendation,
      atlasFlower: flowersByScientificName.get(normalizeKey(recommendation.scientific_name)) ?? null,
    }))
  }, [bouquetInsights, flowersByScientificName])

  useEffect(() => {
    if (!isActive) return

    if (!selectedBouquet || bouquetScientificNames.length === 0) {
      setBouquetInsights(null)
      setBouquetInsightsStatus('idle')
      setBouquetInsightsError(null)
      return
    }

    const cacheKey = `${selectedBouquet.id}:${bouquetScientificNames.join('|')}`
    const cached = bouquetInsightsCacheRef.current.get(cacheKey)
    if (cached) {
      setBouquetInsights(cached)
      setBouquetInsightsStatus('ready')
      setBouquetInsightsError(null)
      return
    }

    let cancelled = false
    setBouquetInsights(null)
    setBouquetInsightsStatus('loading')
    setBouquetInsightsError(null)

    fetch('/api/visualizer-bouquet-insights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scientific_names: bouquetScientificNames }),
    })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((payload: BouquetInsightsResponse) => {
        if (cancelled) return
        bouquetInsightsCacheRef.current.set(cacheKey, payload)
        setBouquetInsights(payload)
        setBouquetInsightsStatus('ready')
        setBouquetInsightsError(null)
      })
      .catch(err => {
        if (cancelled) return
        setBouquetInsightsStatus('error')
        setBouquetInsightsError(
          err instanceof Error
            ? `Could not load bouquet insights: ${err.message}`
            : 'Could not load bouquet insights.',
        )
      })

    return () => { cancelled = true }
  }, [selectedBouquet, bouquetScientificNames, isActive])

  useEffect(() => {
    if (!isActive) return

    if (!flowers.length || introAnimatedRef.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      introAnimatedRef.current = true
      return
    }

    const shell = shellRef.current
    if (!shell) return
    introAnimatedRef.current = true

    const headerItems = shell.querySelectorAll('.viz-eyebrow, .viz-title, .viz-subtitle')
    const stageItems = shell.querySelectorAll('.viz-canvas, .viz-hud__badge, .viz-hud__meta')
    const sidebarCards = shell.querySelectorAll('.viz-sidebar > .viz-info-card')

    const animations = [
      animate(headerItems, {
        opacity: [0, 1],
        translateY: [30, 0],
        filter: ['blur(14px)', 'blur(0px)'],
        delay: stagger(110),
        duration: 1100,
        ease: 'outExpo',
      }),
      animate(stageItems, {
        opacity: [0, 1],
        scale: [0.985, 1],
        translateY: [34, 0],
        delay: stagger(120, { start: 180 }),
        duration: 1200,
        ease: 'outExpo',
      }),
      animate(sidebarCards, {
        opacity: [0, 1],
        translateX: [22, 0],
        translateY: [28, 0],
        delay: stagger(90, { start: 320 }),
        duration: 980,
        ease: 'outCubic',
      }),
    ]

    return () => {
      animations.forEach(animation => animation.revert())
    }
  }, [flowers.length, isActive])

  useEffect(() => {
    if (bouquetInsightsStatus !== 'ready' || !bouquetInsights?.meanings.length) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const list = healthListRef.current
    if (!list) return
    const rows = list.querySelectorAll('.viz-health-row')
    const fills = list.querySelectorAll('.viz-health-row__fill')
    const animations = [
      animate(rows, {
        opacity: [0, 1],
        translateY: [18, 0],
        delay: stagger(80),
        duration: 720,
        ease: 'outCubic',
      }),
      animate(fills, {
        scaleX: [0, 1],
        delay: stagger(90, { start: 100 }),
        duration: 920,
        ease: 'outExpo',
      }),
    ]

    return () => {
      animations.forEach(animation => animation.revert())
    }
  }, [bouquetInsights, bouquetInsightsStatus])

  useEffect(() => {
    if (bouquetInsightsStatus !== 'ready' || !atlasRecommendations.length) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const list = recommendationListRef.current
    if (!list) return
    const cards = list.querySelectorAll('.viz-recommendation-card')
    const thumbs = list.querySelectorAll('.viz-recommendation-card__thumb')
    const animations = [
      animate(cards, {
        opacity: [0, 1],
        translateY: [26, 0],
        scale: [0.96, 1],
        delay: stagger(90),
        duration: 860,
        ease: 'outExpo',
      }),
      animate(thumbs, {
        scale: [0.84, 1],
        rotate: [-4, 0],
        delay: stagger(100, { start: 80 }),
        duration: 820,
        ease: 'outBack',
      }),
    ]

    return () => {
      animations.forEach(animation => animation.revert())
    }
  }, [atlasRecommendations, bouquetInsightsStatus, flowers.length, isActive])

  return (
    <section ref={shellRef} className="viz-shell">
      <header className="viz-header">
        <p className="viz-eyebrow">FloraSense 3D Visualizer</p>
        <h2 className="viz-title">Navigate a Linked Flower Field</h2>
        <p className="viz-subtitle">
          Every flower sits directly in the latent space instead of being packed into bouquet clusters.
          Move through the field with pan and zoom, then click any flower to surface its strongest semantic links, health profile, and recommendations.
        </p>
      </header>

      <div className={`viz-stage${showSidebar ? ' viz-stage--focused' : ''}`}>
        <div className="viz-canvas">
          {flowers.length > 0 && bouquets.length > 0 && !loadError ? (
            <VisualizerCanvas isActive={isActive} flowers={flowers} semanticGraph={semanticGraph} bouquets={bouquets}
              selectedFlowerId={selectedFlowerId} selectedBouquetId={selectedBouquetId}
              viewportResetToken={viewportResetToken}
              onFlowerSelect={handleFlowerSelect} onBouquetSelect={handleBouquetSelect}
              onSceneStatusChange={handleSceneStatus} />
          ) : (
            <div className="viz-canvas__fallback">{loadError ?? 'Loading 3D Visualizer...'}</div>
          )}
          <div className="viz-hud">
            <div className="viz-hud__badge">
              <span className={`viz-panel__status viz-panel__status--${sceneStatus}`}>{sceneStatus}</span>
              <p className="viz-panel__message">{statusMessage}</p>
              <div className="viz-hud__actions">
                <button type="button" className="viz-hud__action" onClick={handleResetFocus}>
                  Reset focus
                </button>
              </div>
            </div>
            <div className="viz-hud__meta">
              <strong>{flowers.length} flowers in the field</strong>
              <span>{semanticGraph.edges.length} visible semantic links across meaning, color, occasion, and latent similarity</span>
            </div>
          </div>
        </div>

        {showSidebar ? (
          <aside className="viz-sidebar">
          <div className="viz-info-card">
            <h3>Bouquet Health Bar</h3>
            {selectedBouquet ? (
              bouquetInsightsStatus === 'loading' ? (
                <p>Calculating the bouquet meaning balance from its selected flowers...</p>
              ) : bouquetInsightsStatus === 'error' ? (
                <p>{bouquetInsightsError ?? 'Bouquet insights failed to load.'}</p>
              ) : bouquetInsights && bouquetInsights.meanings.length > 0 ? (
                <div ref={healthListRef} className="viz-health-list">
                  {bouquetInsights.meanings.map(meaning => (
                    <div key={meaning.label} className="viz-health-row">
                      <div className="viz-health-row__head">
                        <strong>{toTitleCase(meaning.label)}</strong>
                        <span>{toPercent(meaning.score)}</span>
                      </div>
                      <div className="viz-health-row__track" aria-hidden="true">
                        <span className="viz-health-row__fill" style={{ width: toPercent(meaning.score) }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p>This bouquet does not yet have enough shared meaning signals to build a health bar.</p>
              )
            ) : (
              <p>Select a flower to compute the meaning mix of its local neighborhood.</p>
            )}
          </div>

          <div className="viz-info-card">
            <h3>Bouquet Recommendations</h3>
            {selectedBouquet ? (
              bouquetInsightsStatus === 'loading' ? (
                <p>Finding compatible flowers to extend this bouquet...</p>
              ) : bouquetInsightsStatus === 'error' ? (
                <p>{bouquetInsightsError ?? 'Bouquet recommendations failed to load.'}</p>
              ) : atlasRecommendations.length > 0 ? (
                <div ref={recommendationListRef} className="viz-recommendation-list">
                  {atlasRecommendations.map(({ recommendation, atlasFlower }) => (
                    <div key={recommendation.scientific_name} className="viz-recommendation-card">
                      <div className="viz-recommendation-card__head">
                        <div className="viz-recommendation-card__identity">
                          {recommendation.image_url ? (
                            <img
                              className="viz-recommendation-card__thumb"
                              src={recommendation.image_url}
                              alt={recommendation.name}
                            />
                          ) : null}
                          <div>
                            <strong>{recommendation.name}</strong>
                            <span>{recommendation.scientific_name}</span>
                          </div>
                        </div>
                        <em>{toScoreLabel(recommendation.score)}</em>
                      </div>
                      <div className="viz-chip-list">
                        {recommendation.matched_keywords.slice(0, 3).map(term => (
                          <span key={`${recommendation.scientific_name}-${term.keyword}`} className="viz-chip">
                            {term.keyword}
                          </span>
                        ))}
                      </div>
                      <p>
                        {atlasFlower
                          ? 'Already loaded in the 3D Visualizer.'
                          : 'Recommended from the full corpus and not currently rendered in this 3D Visualizer view.'}
                      </p>
                      {atlasFlower ? (
                        <button
                          type="button"
                          className="viz-toolbar__button viz-toolbar__button--compact"
                          onClick={() => handleFlowerSelect(atlasFlower.id)}
                        >
                          Focus in 3D Visualizer
                        </button>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p>No additional flowers were recommended for this neighborhood.</p>
              )
            ) : (
              <p>Select a flower to generate compatible flower recommendations.</p>
            )}
          </div>

          <div className="viz-info-card">
            <h3>Semantic Signals</h3>
            {selectedFlower ? (
              <div className="viz-meta-block">
                {([['Colors', selectedFlower.colors], ['Meanings', selectedFlower.meanings.slice(0, 6)], ['Occasions', selectedFlower.occasions.slice(0, 6)]] as [string, string[]][]).map(([label, vals]) => (
                  <div key={label}>
                    <h4>{label}</h4>
                    <div className="viz-chip-list">{vals.map(v => <span key={v} className="viz-chip">{v}</span>)}</div>
                  </div>
                ))}
              </div>
            ) : <p>Metadata appears here after selecting a flower.</p>}
          </div>
          </aside>
        ) : null}
      </div>
    </section>
  )
}

export default Visualizer3D
