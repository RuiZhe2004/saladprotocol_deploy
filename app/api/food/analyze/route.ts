import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get("image") as File
    const username = formData.get("username") as string

    if (!image || !username) {
      return NextResponse.json({ error: "Image and username are required" }, { status: 400 })
    }

    // Create FormData for Python backend
    const backendFormData = new FormData()
    backendFormData.append("image", image)
    backendFormData.append("username", username)

    // Log the environment variable (for debugging)
    console.log("PYTHON_BACKEND_URL:", process.env.PYTHON_BACKEND_URL);

    // Call Python backend for food analysis
    const backendResponse = await fetch(`${process.env.PYTHON_BACKEND_URL}/food/analyze`, {
      method: "POST",
      body: backendFormData,
    })

    const data = await backendResponse.json()

    if (backendResponse.ok) {
      return NextResponse.json(data)
    } else {
      return NextResponse.json({ error: data.error || "Food analysis failed" }, { status: backendResponse.status })
    }
  } catch (error) {
    console.error("Food analysis API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
