"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import ModelMetrics from "@/components/model-metrics"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function MetricsPage() {
  return (
    <main className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="mb-6">
          <Link href="/">
            <Button variant="outline" className="flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
          </Link>
        </div>

        <h1 className="text-3xl font-bold mb-6">Model Performance Metrics</h1>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Model Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <ModelMetrics />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>About the Models</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="text-lg font-medium mb-2">LSTM Neural Network</h3>
              <p className="text-gray-600">
                Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order
                dependence in sequence prediction problems. This makes them ideal for analyzing text data where context
                and sequence matter. Our LSTM model has been trained on thousands of examples of both fraudulent and
                legitimate phone call messages to identify patterns and indicators of fraud.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2">Naive Bayes Classifier</h3>
              <p className="text-gray-600">
                Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong independence
                assumptions between features. Despite its simplicity, it often performs well in text classification
                tasks. We've included it as a baseline comparison to demonstrate the improved performance of our LSTM
                model.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2">Evaluation Metrics</h3>
              <ul className="list-disc pl-5 text-gray-600 space-y-1">
                <li>
                  <span className="font-medium">Accuracy:</span> The proportion of correctly classified messages (both
                  fraud and legitimate)
                </li>
                <li>
                  <span className="font-medium">Precision:</span> The proportion of messages classified as fraud that
                  are actually fraudulent
                </li>
                <li>
                  <span className="font-medium">Recall:</span> The proportion of actual fraudulent messages that were
                  correctly identified
                </li>
                <li>
                  <span className="font-medium">F1 Score:</span> The harmonic mean of precision and recall, providing a
                  balance between the two
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
