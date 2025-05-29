import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { username } = await request.json()

    if (!username || username.trim().length === 0) {
      return NextResponse.json({ error: "Username is required" }, { status: 400 })
    }

    // Call Python backend to check if user exists
    const backendResponse = await fetch(`${process.env.PYTHON_BACKEND_URL}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username: username.trim() }),
    })

    const data = await backendResponse.json()

    if (backendResponse.ok) {
      return NextResponse.json({
        success: true,
        isNewUser: data.is_new_user,
        user: data.user,
      })
    } else {
      return NextResponse.json({ error: data.error || "Login failed" }, { status: backendResponse.status })
    }
  } catch (error) {
    console.error("Login API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
