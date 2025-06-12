import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { message, username, lastFoodAnalysis, conversationHistory } = await request.json()

    if (!message || !username) {
      return NextResponse.json({ error: "Message and username are required" }, { status: 400 })
    }

    // Call Python backend for chat response
    const backendResponse = await fetch(`${process.env.PYTHON_BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        username,
        last_food_analysis: lastFoodAnalysis,
        conversation_history: conversationHistory
      }),
    })

    const data = await backendResponse.json()

    if (backendResponse.ok) {
      return NextResponse.json({
        response: data.response,
      })
    } else {
      return NextResponse.json({ error: data.error || "Chat failed" }, { status: backendResponse.status })
    }
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
