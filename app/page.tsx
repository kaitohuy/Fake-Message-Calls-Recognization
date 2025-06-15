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
  // Tin nhắn lừa đảo (Fraud)
  "URGENT: Your bank account has been suspended. Verify now!",
  "Congratulations! You've won a free vacation. Click here to claim!",
  "Security Alert: Suspicious login detected. Please reset your password.",
  "Your PayPal account limited! Verify identity: http://paypal-secure.com",
  "You've been selected for $5000 reward! Claim: bit.ly/win5000",
  "IRS Notice: Tax refund pending. Submit SSN now!",
  "Apple ID Locked: Confirm details immediately!",
  "Urgent: Your package delivery failed. Update address: dhl-scam.com",
  "Free iPhone 15! Just pay shipping: apple-freegift.cc",
  "Investment opportunity! Double your money in 24 hours!",
  "Social Security suspended! Call now: 1-888-555-1234",

  // Tin nhắn an toàn (Normal)
  "Hi, just checking if we're still on for tomorrow's meeting.",
  "Reminder: Your service appointment is scheduled for Friday at 3 PM.",
  "Your Amazon order #12345 has shipped. Tracking: https://amazon.com/track",
  "Netflix: Your payment was successful. Thank you!",
  "Happy birthday! Dinner at 7 PM? Let me know if you can make it.",
  "Team meeting moved to 10 AM. Conference room 3B.",
  "Your prescription is ready for pickup at CVS Pharmacy.",
  "Flight reminder: LAX to JFK departs at 8:45 AM tomorrow.",
  "Doctor's appointment confirmed for Monday at 2:30 PM.",
  "Weather alert: Thunderstorms expected this afternoon.",
  "Your car service is complete. Ready for pickup.",

  // Tin nhắn ranh giới (Borderline)
  "Account notification: New login from unknown device",
  "Important security update for your account",
  "Payment confirmation: $49.99 charged to your card",
  "Limited time offer: 50% off all products!",
  "Your subscription will renew automatically",

  // Tin nhắn ngắn/không ngữ cảnh
  "Verify now!",
  "Prize claim",
  "Meeting?",
  "Call me",
  "Congratulations!"
];

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
        confidence: response.confidence,
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
                      Confidence: {(item.confidence * 100).toFixed(1)}%
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