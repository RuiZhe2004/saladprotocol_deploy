import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { message, username, lastFoodAnalysis, conversationHistory } = await request.json()

    if (!message || !username) {
      console.error("Message and username are required");
      return NextResponse.json({ error: "Message and username are required" }, { status: 400 });
    }

    // Construct the backend URL
    const backendUrl = `${process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL}/chat`;

    console.log(`Calling backend at: ${backendUrl}`);

    // Call Python backend for chat response
    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        username,
        last_food_analysis: lastFoodAnalysis,
        conversation_history: conversationHistory,
      }),
    });

    const data = await backendResponse.json();

    console.log("Backend response:", data);

    if (backendResponse.ok) {
      return NextResponse.json({
        response: data.response,
      });
    } else {
      console.error("Backend returned an error:", data);
      return NextResponse.json({ error: data.error || "Chat failed" }, { status: backendResponse.status });
    }
  } catch (error: any) {
    console.error("Chat API error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
