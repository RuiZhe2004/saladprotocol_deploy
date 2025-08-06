import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { username } = await request.json()

    if (!username || username.trim().length === 0) {
      console.error("Username is required")
      return NextResponse.json({ error: "Username is required" }, { status: 400 })
    }

    // Construct the backend URL
    const backendUrl = `${process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL}/auth/login`;

    console.log(`Calling backend at: ${backendUrl}`)

    // Call Python backend to check if user exists
    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username: username.trim() }),
    })

    const data = await backendResponse.json()

    console.log("Backend response:", data)

    if (backendResponse.ok) {
      return NextResponse.json({
        success: true,
        isNewUser: data.is_new_user,
        user: data.user,
      })
    } else {
      console.error("Backend returned an error:", data)
      return NextResponse.json({ error: data.error || "Login failed" }, { status: backendResponse.status })
    }
  } catch (error: any) {
    console.error("Login API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
