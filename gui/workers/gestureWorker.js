const MP_DETECT_INTERVAL_MS = 50

const ROTATION_SENSITIVITY = 120
const ZOOM_SENSITIVITY = 10.0

const VELOCITY_EMA = 0.5
const DEAD_ZONE = 0.015

const CONFIRM_MS = {
  PINCH: 40,
  OPEN_PALM: 50,
  FIST: 180,
  POINT: 100,
  V_SIGN: 300,
  NONE: 120,
}

const PALM_BASE = [0, 5, 9, 13, 17]

let handLandmarker = null
let loading = false
let statsInterval = null

let lastProcessMs = 0
let prevTimeMs = 0
let prevCenter = { x: 0, y: 0, z: 0, has: false }
let vxEma = 0
let vyEma = 0
let vzEma = 0

let rawGesture = "NONE"
let gestureStartMs = 0
let confirmedGesture = "NONE"

let lastFrameMs = 0
let lastHandsCount = 0
let lastError = ""
let initErrorMessage = ""

function classifyGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) return "NONE"

  const wrist = landmarks[0]
  const thumbTip = landmarks[4]
  const indexTip = landmarks[8]
  const palmSize = Math.hypot(
    landmarks[9].x - wrist.x,
    landmarks[9].y - wrist.y
  ) || 0.001
  const pinchDistance = Math.hypot(
    thumbTip.x - indexTip.x,
    thumbTip.y - indexTip.y
  ) / palmSize

  if (pinchDistance < 0.35) return "PINCH"

  const fingers = [
    { tip: landmarks[8], pip: landmarks[6] },
    { tip: landmarks[12], pip: landmarks[10] },
    { tip: landmarks[16], pip: landmarks[14] },
    { tip: landmarks[20], pip: landmarks[18] },
  ]
  const extended = fingers.map(({ tip, pip }) => tip.y < pip.y)
  const extendedCount = extended.filter(Boolean).length

  if (extendedCount === 0) return "FIST"
  if (extendedCount >= 4) return "OPEN_PALM"
  if (extended[0] && extendedCount === 1) return "POINT"
  if (extended[0] && extended[1] && !extended[2] && !extended[3]) return "V_SIGN"

  return "NONE"
}

function resetState() {
  lastProcessMs = 0
  prevTimeMs = 0
  prevCenter.has = false
  vxEma = 0
  vyEma = 0
  vzEma = 0
  rawGesture = "NONE"
  gestureStartMs = 0
  confirmedGesture = "NONE"
}

async function initWorker(wasmUrl, modelUrl) {
  if (handLandmarker || loading) return
  loading = true
  try {
    const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision")
    const vision = await FilesetResolver.forVisionTasks(wasmUrl)
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelUrl,
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    self.postMessage({ type: "ready" })
  } catch (err) {
    initErrorMessage = err instanceof Error ? err.message : "init_failed"
    lastError = "init_failed"
    self.postMessage({ type: "error", message: initErrorMessage })
  } finally {
    loading = false
  }
}

function closeBitmap(bitmap) {
  if (bitmap && typeof bitmap.close === "function") {
    bitmap.close()
  }
}

self.onmessage = (event) => {
  const data = event.data || {}
  if (data.type === "init") {
    initWorker(data.wasmUrl, data.modelUrl)
    if (!statsInterval) {
      statsInterval = setInterval(() => {
        self.postMessage({
          type: "stats",
          lastFrameMs,
          lastHandsCount,
          lastGesture: confirmedGesture,
          lastError,
          initErrorMessage,
        })
      }, 2000)
    }
    return
  }

  if (data.type === "reset") {
    resetState()
    return
  }

  if (data.type !== "frame") return

  if (!handLandmarker) {
    closeBitmap(data.bitmap)
    return
  }

  if (data.nowMs - lastProcessMs < MP_DETECT_INTERVAL_MS) {
    closeBitmap(data.bitmap)
    return
  }

  lastProcessMs = data.nowMs
  lastFrameMs = data.nowMs

  let hands = []

  try {
    const result = handLandmarker.detectForVideo(data.bitmap, data.nowMs)
    if (result.landmarks && result.landmarks.length > 0) {
      hands = result.landmarks.map((lmList, i) => ({
        landmarks: lmList.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
        handedness: (result.handedness[i]?.[0]?.categoryName === "Left" ? "Left" : "Right"),
      }))
    }
    lastError = ""
  } catch {
    lastError = "detect_failed"
    hands = []
  } finally {
    closeBitmap(data.bitmap)
  }

  lastHandsCount = hands.length

  const primaryLandmarks = hands[0]?.landmarks || []
  const gesture = classifyGesture(primaryLandmarks)

  if (primaryLandmarks.length < 21) {
    prevCenter.has = false
    vxEma = 0
    vyEma = 0
    vzEma = 0
    rawGesture = "NONE"
    confirmedGesture = "NONE"
    self.postMessage({
      type: "update",
      rotationDelta: 0,
      scaleDelta: 0,
      gesture: "NONE",
      hands,
      handsDetected: hands.length,
      nowMs: data.nowMs,
      indexTip: null,
    })
    return
  }

  const dt = prevTimeMs
    ? Math.min(Math.max((data.nowMs - prevTimeMs) / 1000, 0.001), 0.1)
    : 0.033
  prevTimeMs = data.nowMs

  if (gesture !== rawGesture) {
    rawGesture = gesture
    gestureStartMs = data.nowMs
  }

  const held = data.nowMs - gestureStartMs
  if (held >= (CONFIRM_MS[gesture] ?? 200) && confirmedGesture !== gesture) {
    vxEma = 0
    vyEma = 0
    prevCenter.has = false
    confirmedGesture = gesture
  }

  let sumX = 0
  let sumY = 0
  let sumZ = 0
  for (let i = 0; i < PALM_BASE.length; i++) {
    const idx = PALM_BASE[i]
    sumX += primaryLandmarks[idx].x
    sumY += primaryLandmarks[idx].y
    sumZ += primaryLandmarks[idx].z ?? 0
  }
  const cx = sumX / PALM_BASE.length
  const cy = sumY / PALM_BASE.length
  const cz = sumZ / PALM_BASE.length

  let vx = 0
  let vy = 0
  let vz = 0
  if (prevCenter.has) {
    const rawVx = (cx - prevCenter.x) / dt
    const rawVy = (cy - prevCenter.y) / dt
    const rawVz = (cz - prevCenter.z) / dt
    vxEma = VELOCITY_EMA * vxEma + (1 - VELOCITY_EMA) * rawVx
    vyEma = VELOCITY_EMA * vyEma + (1 - VELOCITY_EMA) * rawVy
    vzEma = VELOCITY_EMA * vzEma + (1 - VELOCITY_EMA) * rawVz
    vx = Math.abs(vxEma) < DEAD_ZONE ? 0 : vxEma
    vy = Math.abs(vyEma) < DEAD_ZONE ? 0 : vyEma
    vz = Math.abs(vzEma) < DEAD_ZONE ? 0 : vzEma
  }
  prevCenter.x = cx
  prevCenter.y = cy
  prevCenter.z = cz
  prevCenter.has = true

  let rotationDelta = 0
  let scaleDelta = 0

  if (confirmedGesture === "PINCH") {
    rotationDelta = vx * ROTATION_SENSITIVITY * dt
  }

  if (confirmedGesture === "OPEN_PALM") {
    // MediaPipe z grows away from camera; invert so forward motion zooms in.
    scaleDelta = -vz * ZOOM_SENSITIVITY * dt
  }

  const indexTip = primaryLandmarks[8]
    ? { x: primaryLandmarks[8].x, y: primaryLandmarks[8].y }
    : null

  self.postMessage({
    type: "update",
    rotationDelta,
    scaleDelta,
    gesture: confirmedGesture,
    hands,
    handsDetected: hands.length,
    nowMs: data.nowMs,
    indexTip,
  })
}
