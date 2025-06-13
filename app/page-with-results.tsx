"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Shield, Brain } from "lucide-react"
import ResultDisplay from "@/components/result-display"
import { detectFraud } from "@/lib/api"

export default function HomeWithResults() {
  const [message, setMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<null | { isFraud: boolean; confidence: number; message: string }>(null)

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault()
      if (!message.trim()) return

      setIsLoading(true)
      try {
        const response = await detectFraud(message)
        const adjustedResponse = {
          ...response,
          confidence: Math.min(response.confidence, 1) // Giới hạn tối đa 1
        }
        setResult(adjustedResponse)
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

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="flex justify-center gap-2 mb-4">
          <Shield className="h-10 w-10 text-blue-600" />
          <Brain className="h-10 w-10 text-purple-500" />
        </div>

        <h1 className="text-4xl font-bold text-center mb-3">AI Fraud Call Detection</h1>

        <p className="text-center text-gray-600 mb-12 max-w-2xl mx-auto">
          Advanced artificial intelligence system to detect fraudulent phone calls and messages. Powered by LSTM neural
          networks and natural language processing.
        </p>

        <div className="bg-white rounded-lg shadow-sm p-8 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5 text-gray-700"
            >
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
            </svg>
            <h2 className="text-lg font-medium">Message Analysis</h2>
          </div>

          <p className="text-gray-600 text-sm mb-4">
            Enter the phone call message or transcript you want to analyze for potential fraud
          </p>

          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">
                Phone Call Message
              </label>
              <textarea
                id="message"
                className="w-full border border-gray-300 rounded-md p-3 min-h-[100px] focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter the phone call message here... For example: 'This is urgent! Your bank account has been suspended. Please call 1-800-XXX-XXXX immediately to verify your account details.'"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
              ></textarea>
            </div>

            <div className="flex gap-2">
              <Button
                type="submit"
                className="flex-1 bg-gray-500 hover:bg-gray-600 text-white flex items-center justify-center gap-2"
                disabled={isLoading || !message.trim()}
              >
                {isLoading ? (
                  <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></span>
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-5 w-5"
                  >
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                  </svg>
                )}
                Analyze Message
              </Button>
              <Button type="button" variant="outline" onClick={handleClear} className="border-gray-300 text-gray-700">
                Clear
              </Button>
            </div>
          </form>

          {result && <ResultDisplay result={result} />}
        </div>

        <div className="bg-white rounded-lg shadow-sm p-8">
          <h2 className="text-xl font-bold text-blue-600 mb-4">How It Works</h2>

          <p className="text-gray-600 mb-4">
            This AI-powered system uses advanced machine learning techniques including:
          </p>

          <ul className="space-y-3 mb-6">
            <li className="flex items-start">
              <span className="text-blue-600 font-bold mr-2">•</span>
              <div>
                <span className="font-medium text-blue-600">LSTM Neural Networks:</span>
                <span className="text-gray-600"> For understanding context and sequence patterns</span>
              </div>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 font-bold mr-2">•</span>
              <div>
                <span className="font-medium text-blue-600">Natural Language Processing:</span>
                <span className="text-gray-600"> To analyze text content and sentiment</span>
              </div>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 font-bold mr-2">•</span>
              <div>
                <span className="font-medium text-blue-600">Pattern Recognition:</span>
                <span className="text-gray-600"> To identify common fraud indicators and tactics</span>
              </div>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 font-bold mr-2">•</span>
              <div>
                <span className="font-medium text-blue-600">Real-time Analysis:</span>
                <span className="text-gray-600"> Immediate detection and risk assessment</span>
              </div>
            </li>
          </ul>

          <p className="text-blue-600">
            Always verify suspicious calls independently and never provide personal information to unknown callers.
          </p>
        </div>
      </div>
    </main>
  )
}
