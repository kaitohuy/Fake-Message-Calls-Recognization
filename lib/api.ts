// This is a mock API that simulates the backend ML model
// In a real application, this would make a fetch request to a Python Flask backend

// Common fraud keywords and patterns
const fraudKeywords = [
  "urgent",
  "immediate",
  "account suspended",
  "verify your account",
  "bank account",
  "credit card",
  "password",
  "social security",
  "lottery",
  "prize",
  "won",
  "inheritance",
  "money transfer",
  "investment opportunity",
  "cryptocurrency",
  "bitcoin",
  "wallet",
  "suspicious activity",
  "unauthorized",
  "click here",
  "limited time",
  "offer expires",
  "act now",
  "free money",
  "guaranteed return",
]

export async function detectFraud(message: string): Promise<{ isFraud: boolean; confidence: number; message: string }> {
  try {
    // Make a POST request to your Flask backend
    const response = await fetch("http://localhost:5000/api/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`)
    }

    // Parse the JSON response
    const data = await response.json()

    return {
      isFraud: data.isFraud,
      confidence: data.confidence,
      message: message,
    }
  } catch (error) {
    console.error("Error calling fraud detection API:", error)

    // Fallback to client-side detection if API call fails
    const messageLower = message.toLowerCase()

    // Count keyword matches
    const keywordMatches = fraudKeywords.filter((keyword) => messageLower.includes(keyword.toLowerCase())).length
    const keywordRatio = keywordMatches / fraudKeywords.length

    // Additional heuristics
    const hasUrgency = /urgent|immediately|today|now|hurry|quick/i.test(messageLower)
    const hasMoneyMention = /\$|\bmoney\b|\bpayment\b|\bfund\b|\bcash\b/i.test(messageLower)
    const hasPersonalInfoRequest = /\bpassword\b|\bsocial security\b|\bpin\b|\baccount\b|\bverify\b/i.test(messageLower)

    // Combine factors for final decision
    const riskFactors = [
      keywordRatio * 0.5, // Weight keyword matches
      hasUrgency ? 0.2 : 0,
      hasMoneyMention ? 0.15 : 0,
      hasPersonalInfoRequest ? 0.25 : 0,
    ]

    const totalRiskScore = riskFactors.reduce((sum, score) => sum + score, 0)
    const normalizedScore = Math.min(totalRiskScore, 1) // Cap at 1.0

    // Determine if it's fraud based on threshold
    const isFraud = normalizedScore > 0.3

    // Add some randomness to confidence for demo purposes
    const randomFactor = 0.05 * (Math.random() - 0.5)
    const confidence = Math.max(
      0,
      Math.min(1, isFraud ? 0.5 + normalizedScore * 0.5 + randomFactor : 1 - normalizedScore + randomFactor),
    )

    return {
      isFraud,
      confidence,
      message,
    }
  }
}
