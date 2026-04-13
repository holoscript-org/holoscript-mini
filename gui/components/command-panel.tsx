"use client"

import { useState, useRef, useEffect } from "react"
import { HoloPanel } from "./holo-panel"
import { Send, Mic, MicOff, Terminal } from "lucide-react"

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
      <div className="flex flex-col h-full min-h-[200px]">
        {/* Command history */}
        <div
          ref={historyRef}
          className="flex-1 overflow-auto mb-3 p-2 rounded bg-background/50 border border-primary/10"
          style={{ maxHeight: "150px" }}
        >
          {commandHistory.length === 0 ? (
            <div className="text-center text-muted-foreground text-xs font-mono py-4">
              No commands yet
            </div>
          ) : (
            <div className="space-y-1">
              {commandHistory.map((entry) => (
                <div
                  key={entry.id}
                  className={`flex items-start gap-2 text-xs font-mono ${
                    entry.type === "user"
                      ? "text-primary"
                      : "text-muted-foreground"
                  }`}
                >
                  <span className="text-muted-foreground/50 shrink-0">
                    [{mounted ? formatTime(entry.timestamp) : "--:--:--"}]
                  </span>
                  <span
                    className={`shrink-0 ${
                      entry.type === "user" ? "text-primary" : "text-chart-2"
                    }`}
                  >
                    {entry.type === "user" ? ">" : "$"}
                  </span>
                  <span className="break-all">{entry.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <Terminal className="w-3 h-3" />
            Describe simulation
          </label>

          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSendCommand()
                  }
                }}
                placeholder="Enter command..."
                className="w-full px-3 py-2 text-sm font-mono bg-input border border-primary/30 rounded text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/50 transition-colors"
              />
              {isListening && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  <span className="flex items-center gap-1 text-xs text-primary animate-pulse">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                    Listening...
                  </span>
                </div>
              )}
            </div>

            <button
              onClick={toggleVoiceInput}
              className={`p-2 rounded border transition-colors ${
                isListening
                  ? "border-primary bg-primary/20 text-primary animate-pulse"
                  : "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20"
              }`}
              aria-label={isListening ? "Stop listening" : "Start voice input"}
            >
              {isListening ? (
                <MicOff className="w-4 h-4" />
              ) : (
                <Mic className="w-4 h-4" />
              )}
            </button>

            <button
              onClick={handleSendCommand}
              disabled={!inputValue.trim()}
              className="p-2 rounded border border-primary/30 bg-primary/10 text-primary hover:bg-primary/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Send command"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </HoloPanel>
  )
}
