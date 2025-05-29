"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Leaf, Salad } from "lucide-react"

export default function LoginPage() {
  const [username, setUsername] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim()) return

    setIsLoading(true)

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: username.trim() }),
      })

      const data = await response.json()

      if (response.ok) {
        localStorage.setItem("username", username.trim())

        if (data.isNewUser) {
          router.push("/profile-setup")
        } else {
          router.push("/chat")
        }
      } else {
        alert("Login failed. Please try again.")
      }
    } catch (error) {
      console.error("Login error:", error)
      alert("An error occurred. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-green-500 p-3 rounded-full">
              <Salad className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-green-800 mb-2">Salad Protocol</h1>
          <p className="text-green-600">Your AI Nutritionist Companion</p>
        </div>

        {/* Login Card */}
        <Card className="shadow-lg border-green-200">
          <CardHeader className="text-center">
            <CardTitle className="text-green-700 flex items-center justify-center gap-2">
              <Leaf className="h-5 w-5" />
              Welcome Back
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username" className="text-green-700">
                  Username
                </Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="border-green-300 focus:border-green-500 focus:ring-green-500"
                  required
                />
              </div>

              <Button
                type="submit"
                className="w-full bg-green-600 hover:bg-green-700 text-white"
                disabled={isLoading || !username.trim()}
              >
                {isLoading ? "Signing In..." : "Sign In"}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-green-600">New to Salad Protocol? Just enter a username to get started!</p>
            </div>
          </CardContent>
        </Card>

        {/* Features Preview */}
        <div className="mt-8 text-center">
          <div className="grid grid-cols-2 gap-4 text-sm text-green-700">
            <div className="bg-white/50 p-3 rounded-lg">
              <div className="font-semibold">🥗 Nutrition Analysis</div>
              <div className="text-xs">AI-powered meal insights</div>
            </div>
            <div className="bg-white/50 p-3 rounded-lg">
              <div className="font-semibold">📸 Food Recognition</div>
              <div className="text-xs">Upload & analyze meals</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
