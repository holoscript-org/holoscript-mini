"use client"

import { cn } from "@/lib/utils"

interface HoloPanelProps {
  title: string
  icon?: React.ReactNode
  className?: string
  children: React.ReactNode
  statusIndicator?: boolean
}

export function HoloPanel({ title, icon, className, children, statusIndicator }: HoloPanelProps) {
  return (
    <div
      className={cn(
        "relative flex flex-col rounded-lg border border-primary/30 bg-card/90 overflow-hidden",
        "before:absolute before:inset-0 before:rounded-lg before:border before:border-primary/20 before:pointer-events-none",
        "shadow-[0_0_8px_rgba(0,255,255,0.08)]",
        className
      )}
    >
      {/* Corner accents */}
      <div className="absolute top-0 left-0 w-3 h-3 border-l-2 border-t-2 border-primary" />
      <div className="absolute top-0 right-0 w-3 h-3 border-r-2 border-t-2 border-primary" />
      <div className="absolute bottom-0 left-0 w-3 h-3 border-l-2 border-b-2 border-primary" />
      <div className="absolute bottom-0 right-0 w-3 h-3 border-r-2 border-b-2 border-primary" />

      {/* Header */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-b border-primary/20">
        {statusIndicator && (
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(0,255,255,0.8)]" />
        )}
        {icon}
        <h2 className="text-sm font-mono text-primary tracking-wider uppercase">{title}</h2>
      </div>

      {/* Content — flex-1 + min-h-0 so children that use h-full actually get height */}
      <div className="flex-1 min-h-0 p-3">{children}</div>
    </div>
  )
}
