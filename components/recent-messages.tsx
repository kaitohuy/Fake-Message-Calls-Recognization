import { AlertCircle, CheckCircle } from "lucide-react"

interface Message {
  text: string
  isFraud: boolean
  timestamp: string
}

interface RecentMessagesProps {
  messages: Message[]
}

export default function RecentMessages({ messages }: RecentMessagesProps) {
  if (messages.length === 0) {
    return <div className="text-center py-8 text-gray-500">No messages analyzed yet</div>
  }

  return (
    <div className="space-y-3 max-h-[400px] overflow-y-auto">
      {messages.map((message, index) => (
        <div key={index} className="p-3 border rounded-lg flex items-start gap-3">
          {message.isFraud ? (
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
          ) : (
            <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <p className="text-sm truncate">{message.text}</p>
            <div className="flex justify-between items-center mt-1">
              <span className={`text-xs font-medium ${message.isFraud ? "text-red-500" : "text-green-500"}`}>
                {message.isFraud ? "Fraud" : "Normal"}
              </span>
              <span className="text-xs text-gray-400">{message.timestamp}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
