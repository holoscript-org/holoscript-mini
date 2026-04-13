"use client"

import { useEffect, useState, useRef } from "react"
import { HoloPanel } from "./holo-panel"
import { Mic } from "lucide-react"

const voiceHistory = [
  "Show me a spinning sphere",
  "Add a glowing ring around it",
  "Make the ring pulse with blue light",
  "Explode the scene gently",
]

export function VoiceConsole() {
  const [isListening, setIsListening] = useState(true)
  const [currentCommand, setCurrentCommand] = useState("Reset and show a galaxy cluster")
  const [waveformData, setWaveformData] = useState<number[]>(Array(50).fill(0))
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const interval = setInterval(() => {
      setWaveformData((prev) => {
        const newData = [...prev.slice(1)]
        newData.push(isListening ? Math.random() * 0.8 + 0.2 : 0.1)
        return newData
      })
    }, 50)

    return () => clearInterval(interval)
  }, [isListening])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    // Draw waveform
    ctx.beginPath()
    ctx.strokeStyle = "#00ffff"
    ctx.lineWidth = 2

    const midY = height / 2

    waveformData.forEach((value, i) => {
      const x = (i / waveformData.length) * width
      const amplitude = value * (height / 2 - 4)
      const y = midY + Math.sin(i * 0.3 + Date.now() * 0.005) * amplitude

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()

    // Add glow effect
    ctx.shadowBlur = 10
    ctx.shadowColor = "#00ffff"
    ctx.stroke()
  }, [waveformData])

  return (
    <HoloPanel title="Voice Console" statusIndicator className="w-full">
      <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
        {/* Microphone and waveform */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => setIsListening(!isListening)}
            className={`p-3 rounded-full border transition-all ${
              isListening
                ? "border-primary bg-primary/20 text-primary shadow-[0_0_15px_rgba(0,255,255,0.5)]"
                : "border-muted bg-muted/20 text-muted-foreground"
            }`}
            aria-label={isListening ? "Stop listening" : "Start listening"}
          >
            <Mic className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-2">
            <span className="text-sm font-mono text-muted-foreground">
              {isListening ? "Listening" : "Paused"}
            </span>
            <span
              className={`w-2 h-2 rounded-full ${
                isListening
                  ? "bg-primary animate-pulse shadow-[0_0_8px_rgba(0,255,255,0.8)]"
                  : "bg-muted"
              }`}
            />
          </div>

          <canvas
            ref={canvasRef}
            width={100}
            height={30}
            className="rounded bg-background/50 border border-primary/10"
          />
        </div>

        {/* Current command */}
        <div className="flex-1">
          <div className="text-xs text-muted-foreground font-mono mb-1">Latest:</div>
          <div className="text-primary font-mono text-sm bg-background/50 px-3 py-2 rounded border border-primary/20">
            [ {currentCommand} ]
          </div>
        </div>

        {/* History */}
        <div className="flex-1 hidden lg:block">
          <div className="text-xs text-muted-foreground font-mono mb-1">History:</div>
          <div className="flex flex-wrap gap-2">
            {voiceHistory.slice(-4).map((cmd, i) => (
              <button
                key={i}
                onClick={() => setCurrentCommand(cmd)}
                className="text-xs font-mono text-muted-foreground hover:text-primary px-2 py-1 rounded bg-background/30 border border-primary/10 hover:border-primary/30 transition-colors truncate max-w-[150px]"
              >
                {cmd}
              </button>
            ))}
          </div>
        </div>
      </div>
    </HoloPanel>
  )
}
