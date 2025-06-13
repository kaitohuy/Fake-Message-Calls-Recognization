"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface ModelMetrics {
  lstm: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
    trainingAccuracy: number
    validationAccuracy: number
    loss: number
  }
  naiveBayes: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
    trainingAccuracy: number
    validationAccuracy: number
    loss: number
  }
  modelLoaded: boolean
}

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const response = await fetch("http://localhost:5000/api/model-info")
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`)
        }
        const data = await response.json()
        setMetrics(data)
      } catch (err) {
        console.error("Error fetching model metrics:", err)
        setError("Failed to load model metrics. Using fallback data.")
        // Use fallback data
        setMetrics({
          modelLoaded: false,
          lstm: {
            accuracy: 0.972,
            precision: 0.968,
            recall: 0.981,
            f1Score: 0.974,
            trainingAccuracy: 0.9989,
            validationAccuracy: 0.972,
            loss: 0.0059,
          },
          naiveBayes: {
            accuracy: 0.891,
            precision: 0.875,
            recall: 0.862,
            f1Score: 0.868,
            trainingAccuracy: 0.9215,
            validationAccuracy: 0.891,
            loss: 0.1106,
          },
        })
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
  }, [])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No metrics data available</p>
      </div>
    )
  }

  const performanceData = [
    {
      name: "Accuracy",
      LSTM: metrics.lstm.accuracy,
      NaiveBayes: metrics.naiveBayes.accuracy,
    },
    {
      name: "Precision",
      LSTM: metrics.lstm.precision,
      NaiveBayes: metrics.naiveBayes.precision,
    },
    {
      name: "Recall",
      LSTM: metrics.lstm.recall,
      NaiveBayes: metrics.naiveBayes.recall,
    },
    {
      name: "F1 Score",
      LSTM: metrics.lstm.f1Score,
      NaiveBayes: metrics.naiveBayes.f1Score,
    },
  ]

  return (
    <div className="space-y-4">
      {error && <div className="text-amber-600 text-sm mb-2">{error}</div>}

      {!metrics.modelLoaded && (
        <div className="text-amber-600 text-sm mb-2">Model not loaded on server. Using sample metrics data.</div>
      )}

      <Tabs defaultValue="comparison">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
          <TabsTrigger value="details">Detailed Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="comparison">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={performanceData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value: number) => [(value * 100).toFixed(1) + "%"]} />
                <Legend />
                <Bar dataKey="LSTM" fill="#3b82f6" name="LSTM" />
                <Bar dataKey="NaiveBayes" fill="#f97316" name="Naive Bayes" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="details">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">LSTM Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Training Accuracy:</div>
                    <div className="text-sm text-right">{(metrics.lstm.trainingAccuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Validation Accuracy:</div>
                    <div className="text-sm text-right">{(metrics.lstm.validationAccuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Loss:</div>
                    <div className="text-sm text-right">{metrics.lstm.loss.toFixed(4)}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Precision:</div>
                    <div className="text-sm text-right">{(metrics.lstm.precision * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Recall:</div>
                    <div className="text-sm text-right">{(metrics.lstm.recall * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">F1 Score:</div>
                    <div className="text-sm text-right">{(metrics.lstm.f1Score * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Naive Bayes Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Training Accuracy:</div>
                    <div className="text-sm text-right">{(metrics.naiveBayes.trainingAccuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Validation Accuracy:</div>
                    <div className="text-sm text-right">
                      {(metrics.naiveBayes.validationAccuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Loss:</div>
                    <div className="text-sm text-right">{metrics.naiveBayes.loss.toFixed(4)}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Precision:</div>
                    <div className="text-sm text-right">{(metrics.naiveBayes.precision * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">Recall:</div>
                    <div className="text-sm text-right">{(metrics.naiveBayes.recall * 100).toFixed(2)}%</div>
                  </div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="text-sm font-medium">F1 Score:</div>
                    <div className="text-sm text-right">{(metrics.naiveBayes.f1Score * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
