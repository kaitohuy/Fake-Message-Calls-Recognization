// app/page.tsx
"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Shield, Brain } from "lucide-react"
import ResultDisplay from "@/components/result-display"
import { detectFraud } from "@/lib/api"

// Danh sách mẫu để random
const SAMPLE_MESSAGES = [
  "URGENT: Your bank account has been suspended. Verify now!",
  "Hi, just checking if we're still on for tomorrow's meeting.",
  "Congratulations! You've won a free vacation. Click here to claim!",
  "Reminder: Your service appointment is scheduled for Friday at 3 PM.",
  "Security Alert: Suspicious login detected. Please reset your password."
]

// Kiểu dữ liệu cho một mục lịch sử
type HistoryItem = {
  message: string
  isFraud: boolean
  confidence: number
}

export default function Home() {
  const [message, setMessage] = useState("")
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<HistoryItem | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const text = message.trim()
    if (!text) return

    setIsLoading(true)
    try {
      const response = await detectFraud(text)
      const item: HistoryItem = {
        message: text,
        isFraud: response.isFraud,
        confidence: Math.min(response.confidence * 100, 100), // Giới hạn tối đa 100%
      }

      setResult(item)
      setHistory((prev) => [item, ...prev])
    } catch (error) {
      console.error("Error detecting fraud:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setMessage("")
    setResult(null)
  }

  const handleRandom = () => {
    const rand =
      SAMPLE_MESSAGES[Math.floor(Math.random() * SAMPLE_MESSAGES.length)]
    setMessage(rand)
    setResult(null)
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12 max-w-4xl">

        {/* ICONS */}
        <div className="flex justify-center gap-2 mb-4">
          <Shield className="h-10 w-10 text-blue-600" />
          <Brain className="h-10 w-10 text-purple-500" />
        </div>
        <h1 className="text-4xl font-bold text-center mb-3">AI Fraud Call Detection</h1>
        <p className="text-center text-gray-600 mb-12 max-w-2xl mx-auto">
          Advanced AI system to detect fraudulent phone calls and messages.
        </p>

        {/* FORM */}
        <form
          onSubmit={handleSubmit}
          className="bg-white rounded-lg shadow-sm p-8 mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium">Message Analysis</h2>
            <Button
              type="button"
              variant="outline"
              onClick={handleRandom}
              className="text-sm"
            >
              Random Message
            </Button>
          </div>

          <textarea
            id="message"
            className="w-full border border-gray-300 rounded-md p-3 min-h-[100px] mb-4 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter the phone call message here..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />

          <div className="flex gap-2 mb-4">
            <Button
              type="submit"
              disabled={isLoading || !message.trim()}
              className="flex-1 bg-gray-500 hover:bg-gray-600 text-white"
            >
              {isLoading ? "Analyzing…" : "Analyze Message"}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={handleClear}
              className="border-gray-300 text-gray-700"
            >
              Clear
            </Button>
          </div>

          {result && <ResultDisplay result={result} />}
        </form>

        {/* HISTORY */}
        {history.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-8">
            <h2 className="text-xl font-bold mb-4">Analysis History</h2>
            <ul className="space-y-4">
              {history.map((item, idx) => (
                <li
                  key={idx}
                  className="flex items-center justify-between border-b pb-2"
                >
                  <div>
                    <p className="text-gray-800">{item.message}</p>
                    <p className="text-sm text-gray-500">
                      Confidence: {item.confidence.toFixed(1)}%
                    </p>
                  </div>
                  <Badge
                    variant={item.isFraud ? "destructive" : "outline"}
                    className="uppercase text-xs"
                  >
                    {item.isFraud ? "Fraud" : "Safe"}
                  </Badge>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  )
}