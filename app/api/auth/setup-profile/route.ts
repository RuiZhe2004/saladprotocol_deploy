import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { username, birthday, height, weight } = await request.json()

    if (!username || !birthday || !height || !weight) {
      console.error("Missing required fields in request")
      return NextResponse.json({ error: "All fields are required" }, { status: 400 })
    }

    // Construct the backend URL
    const backendUrl = `${process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL}/auth/setup-profile`;

    console.log(`Calling backend at: ${backendUrl}`)  // Log the URL

    // Call Python backend to save user profile
    const backendResponse = await fetch(backendUrl, {
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

    console.log("Backend response:", data) // Log the entire response

    if (backendResponse.ok) {
      return NextResponse.json({
        success: true,
        user: data.user,
      })
    } else {
      console.error("Backend returned an error:", data.error || "Profile setup failed")
      return NextResponse.json({ error: data.error || "Profile setup failed" }, { status: backendResponse.status })
    }
  } catch (error: any) {
    console.error("Profile setup API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}