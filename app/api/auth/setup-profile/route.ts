import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { username, birthday, height, weight } = await request.json()

    if (!username || !birthday || !height || !weight) {
      return NextResponse.json({ error: "All fields are required" }, { status: 400 })
    }

    // Call Python backend to save user profile
    const backendResponse = await fetch(`${process.env.PYTHON_BACKEND_URL}/auth/setup-profile`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        username,
        birthday,
        height: Number.parseFloat(height),
        weight: Number.parseFloat(weight),
      }),
    })

    const data = await backendResponse.json()

    if (backendResponse.ok) {
      return NextResponse.json({
        success: true,
        user: data.user,
      })
    } else {
      return NextResponse.json({ error: data.error || "Profile setup failed" }, { status: backendResponse.status })
    }
  } catch (error) {
    console.error("Profile setup API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
