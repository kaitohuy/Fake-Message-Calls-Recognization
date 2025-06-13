import { AlertCircle, CheckCircle } from "lucide-react"

interface ResultDisplayProps {
  result: {
    isFraud: boolean
    confidence: number
    message: string
  } | null
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  if (!result) return null

  const displayConfidence = Math.min(result.confidence * 100, 100).toFixed(1) // Giới hạn tối đa 100%

  return (
    <div className="mt-6 p-4 border rounded-lg">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Analysis Result</h3>
        <span
          className={`text-xs px-3 py-1 rounded-full font-medium ${
            result.isFraud ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"
          }`}
        >
          {result.isFraud ? "FRAUD DETECTED" : "LEGITIMATE MESSAGE"}
        </span>
      </div>

      <div className="mt-4 flex items-start gap-3">
        {result.isFraud ? (
          <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
        ) : (
          <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
        )}
        <div>
          <p className="text-gray-700">
            {result.isFraud
              ? "This message appears to be fraudulent. Exercise caution and do not share any personal information or follow any instructions provided."
              : "This message appears to be legitimate. However, always exercise caution when sharing personal information."}
          </p>
          <p className="text-sm text-gray-500 mt-2">Confidence: {displayConfidence}%</p>
        </div>
      </div>
    </div>
  )
}