import { useState, useEffect } from 'react'
import { X, Copy, Check, Code2, Eye } from 'lucide-react'
import type { Artifact } from '../lib/artifacts'

interface CanvasPanelProps {
  artifact: Artifact | null
  onClose: () => void
}

export function CanvasPanel({ artifact, onClose }: CanvasPanelProps) {
  const [tab, setTab] = useState<'preview' | 'code'>('preview')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    setTab(artifact?.isMarkup ? 'preview' : 'code')
    setCopied(false)
  }, [artifact])

  if (!artifact) return null

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(artifact.code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }

  const tabBtn = (active: boolean) =>
    `px-2.5 py-1 text-xs flex items-center gap-1 transition-colors ${
      active ? 'bg-cyan-500/20 text-cyan-300' : 'text-text-muted hover:text-text'
    }`

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[48%] max-w-3xl bg-surface border-l border-border/40 shadow-2xl flex flex-col animate-in slide-in-from-right">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/40 flex-shrink-0">
        <span className="font-medium text-text">{artifact.title}</span>
        {artifact.language && artifact.language !== 'text' && (
          <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-surface-2 text-text-muted">
            {artifact.language}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          {artifact.isMarkup && (
            <div className="flex rounded-lg overflow-hidden border border-border/40 mr-1">
              <button onClick={() => setTab('preview')} className={tabBtn(tab === 'preview')}>
                <Eye size={13} /> Preview
              </button>
              <button onClick={() => setTab('code')} className={tabBtn(tab === 'code')}>
                <Code2 size={13} /> Code
              </button>
            </div>
          )}
          <button
            onClick={copy}
            className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
            title="Copy code"
          >
            {copied ? <Check size={16} /> : <Copy size={16} />}
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
            title="Close canvas"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden">
        {artifact.isMarkup && tab === 'preview' ? (
          <iframe
            title="Artifact preview"
            srcDoc={artifact.code}
            sandbox="allow-scripts allow-modals allow-popups"
            className="w-full h-full bg-white border-0"
          />
        ) : (
          <pre className="w-full h-full overflow-auto p-4 text-sm font-mono text-text/90 bg-background whitespace-pre">
            {artifact.code}
          </pre>
        )}
      </div>
    </div>
  )
}
