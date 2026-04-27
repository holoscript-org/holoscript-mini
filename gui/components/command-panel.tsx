"use client"

import { useState, useRef, useEffect } from "react"
import { HoloPanel } from "./holo-panel"
import { Send, Mic, MicOff } from "lucide-react"

interface CommandEntry {
  id: string
  text: string
  timestamp: Date
  type: "user" | "system"
}

export function CommandPanel({
  onCommandSend,
}: {
  onCommandSend?: (command: string) => void
}) {
  const [inputValue, setInputValue] = useState("")
  const [commandHistory, setCommandHistory] = useState<CommandEntry[]>([
    {
      id: "1",
      text: "System initialized",
      timestamp: new Date(),
      type: "system",
    },
    {
      id: "2",
      text: "Hologram renderer active",
      timestamp: new Date(),
      type: "system",
    },
  ])
  const [isListening, setIsListening] = useState(false)
  const [mounted, setMounted] = useState(false)
  const historyRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Auto-scroll to bottom when new commands are added
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight
    }
  }, [commandHistory])

  // Setup speech recognition
  useEffect(() => {
    if (typeof window !== "undefined" && "webkitSpeechRecognition" in window) {
      const SpeechRecognition = window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = false

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setInputValue(transcript)
        setIsListening(false)
      }

      recognitionRef.current.onerror = () => {
        setIsListening(false)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }
  }, [])

  const handleSendCommand = () => {
    if (!inputValue.trim()) return

    const newCommand: CommandEntry = {
      id: Date.now().toString(),
      text: inputValue.trim(),
      timestamp: new Date(),
      type: "user",
    }

    setCommandHistory((prev) => [...prev, newCommand])

    // Notify parent
    if (onCommandSend) {
      onCommandSend(inputValue.trim())
    }

    // Simulate system response
    setTimeout(() => {
      const response: CommandEntry = {
        id: (Date.now() + 1).toString(),
        text: `Processing: "${inputValue.trim()}"`,
        timestamp: new Date(),
        type: "system",
      }
      setCommandHistory((prev) => [...prev, response])
    }, 500)

    setInputValue("")
  }

  const toggleVoiceInput = () => {
    if (!recognitionRef.current) {
      // Fallback: simulate voice input
      setIsListening(true)
      setTimeout(() => {
        setInputValue("Show me a spinning cube")
        setIsListening(false)
      }, 1500)
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    } else {
      recognitionRef.current.start()
      setIsListening(true)
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    })
  }

  return (
    <HoloPanel title="Command Interface" statusIndicator className="h-full">
      <div className="flex flex-col h-full min-h-0 gap-2">
        {/* Command history */}
        <div
          ref={historyRef}
          className="flex-1 min-h-0 overflow-auto rounded bg-black/40 border border-primary/15 p-2"
        >
          {commandHistory.length === 0 ? (
            <div className="text-center text-muted-foreground/50 text-xs font-mono py-3">
              — no output yet —
            </div>
          ) : (
            <div className="space-y-0.5">
              {commandHistory.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start gap-1.5 text-xs font-mono leading-5"
                >
                  <span className="text-primary/30 shrink-0 tabular-nums">
                    {mounted ? formatTime(entry.timestamp) : "--:--:--"}
                  </span>
                  <span className={`shrink-0 font-bold ${entry.type === "user" ? "text-primary" : "text-emerald-400"}`}>
                    {entry.type === "user" ? "›" : "»"}
                  </span>
                  <span className={`break-all ${entry.type === "user" ? "text-primary/90" : "text-slate-300"}`}>
                    {entry.text}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input row */}
        <div className="shrink-0 flex items-center gap-2">
          {/* Prompt glyph */}
          <span className="text-primary/50 font-mono text-sm select-none">›_</span>

          <div className="flex-1 relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSendCommand() }}
              placeholder={isListening ? "" : "Enter command or describe a scene…"}
              className="w-full px-3 py-1.5 text-xs font-mono bg-black/40 border border-primary/30 rounded text-primary placeholder:text-muted-foreground/40 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/40 transition-colors"
            />
            {isListening && (
              <div className="absolute inset-0 flex items-center px-3 pointer-events-none">
                <span className="flex items-center gap-1.5 text-xs font-mono text-primary animate-pulse">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                  Listening…
                </span>
              </div>
            )}
          </div>

          {/* Mic toggle */}
          <button
            onClick={toggleVoiceInput}
            className={`p-1.5 rounded border transition-colors ${
              isListening
                ? "border-primary bg-primary/20 text-primary animate-pulse"
                : "border-primary/30 bg-primary/5 text-primary/60 hover:text-primary hover:bg-primary/15"
            }`}
            aria-label={isListening ? "Stop listening" : "Voice input"}
          >
            {isListening ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
          </button>

          {/* Send */}
          <button
            onClick={handleSendCommand}
            disabled={!inputValue.trim()}
            className="p-1.5 rounded border border-primary/30 bg-primary/10 text-primary hover:bg-primary/25 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            aria-label="Send"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </HoloPanel>
  )
}
