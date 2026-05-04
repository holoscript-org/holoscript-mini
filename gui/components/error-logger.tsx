"use client"

import { useEffect } from "react"

export function ErrorLogger() {
  useEffect(() => {
    const onError = (event: ErrorEvent) => {
      const message = event.message || "Unknown error"
      const filename = event.filename || ""
      const location = `${filename}:${event.lineno ?? 0}:${event.colno ?? 0}`
      console.error("[BrowserError]", message, location, event.error)
    }

    const onRejection = (event: PromiseRejectionEvent) => {
      console.error("[BrowserError] Unhandled promise rejection", event.reason)
    }

    window.addEventListener("error", onError)
    window.addEventListener("unhandledrejection", onRejection)

    return () => {
      window.removeEventListener("error", onError)
      window.removeEventListener("unhandledrejection", onRejection)
    }
  }, [])

  return null
}
