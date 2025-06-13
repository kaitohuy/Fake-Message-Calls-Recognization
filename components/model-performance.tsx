"use client"

import { Card } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

const performanceData = [
  {
    name: "Accuracy",
    LSTM: 0.972,
    NaiveBayes: 0.891,
  },
  {
    name: "Precision",
    LSTM: 0.968,
    NaiveBayes: 0.875,
  },
  {
    name: "Recall",
    LSTM: 0.981,
    NaiveBayes: 0.862,
  },
  {
    name: "F1 Score",
    LSTM: 0.974,
    NaiveBayes: 0.868,
  },
]

export default function ModelPerformance() {
  return (
    <div className="space-y-4">
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

      <div className="grid grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="font-medium text-sm mb-2">LSTM Model</h3>
          <p className="text-xs text-gray-500">
            Training Accuracy: 99.89%
            <br />
            Validation Accuracy: 97.20%
            <br />
            Loss: 0.0059
          </p>
        </Card>
        <Card className="p-4">
          <h3 className="font-medium text-sm mb-2">Naive Bayes</h3>
          <p className="text-xs text-gray-500">
            Training Accuracy: 92.15%
            <br />
            Validation Accuracy: 89.10%
            <br />
            Loss: 0.1106
          </p>
        </Card>
      </div>
    </div>
  )
}
