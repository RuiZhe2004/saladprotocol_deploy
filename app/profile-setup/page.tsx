"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Leaf, User, Calendar, Ruler, Weight } from "lucide-react"

export default function ProfileSetup() {
  const [formData, setFormData] = useState({
    birthday: "",
    height: "",
    weight: "",
  })
  const [isLoading, setIsLoading] = useState(false)
  const [username, setUsername] = useState("")
  const router = useRouter()

  useEffect(() => {
    const storedUsername = localStorage.getItem("username")
    if (!storedUsername) {
      router.push("/")
      return
    }
    setUsername(storedUsername)
  }, [router])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const response = await fetch("/api/auth/setup-profile", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          ...formData,
        }),
      })

      if (response.ok) {
        router.push("/chat")
      } else {
        alert("Failed to save profile. Please try again.")
      }
    } catch (error) {
      console.error("Profile setup error:", error)
      alert("An error occurred. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-green-500 p-3 rounded-full">
              <User className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-green-800 mb-2">Complete Your Profile</h1>
          <p className="text-green-600">Help us personalize your nutrition journey</p>
        </div>

        {/* Profile Setup Card */}
        <Card className="shadow-lg border-green-200">
          <CardHeader className="text-center">
            <CardTitle className="text-green-700 flex items-center justify-center gap-2">
              <Leaf className="h-5 w-5" />
              Welcome, {username}!
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="birthday" className="text-green-700 flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Birthday
                </Label>
                <Input
                  id="birthday"
                  name="birthday"
                  type="date"
                  value={formData.birthday}
                  onChange={handleInputChange}
                  className="border-green-300 focus:border-green-500 focus:ring-green-500"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="height" className="text-green-700 flex items-center gap-2">
                  <Ruler className="h-4 w-4" />
                  Height (cm)
                </Label>
                <Input
                  id="height"
                  name="height"
                  type="number"
                  placeholder="e.g., 170"
                  value={formData.height}
                  onChange={handleInputChange}
                  className="border-green-300 focus:border-green-500 focus:ring-green-500"
                  min="100"
                  max="250"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="weight" className="text-green-700 flex items-center gap-2">
                  <Weight className="h-4 w-4" />
                  Weight (kg)
                </Label>
                <Input
                  id="weight"
                  name="weight"
                  type="number"
                  placeholder="e.g., 70"
                  value={formData.weight}
                  onChange={handleInputChange}
                  className="border-green-300 focus:border-green-500 focus:ring-green-500"
                  min="30"
                  max="300"
                  step="0.1"
                  required
                />
              </div>

              <Button type="submit" className="w-full bg-green-600 hover:bg-green-700 text-white" disabled={isLoading}>
                {isLoading ? "Setting Up..." : "Complete Setup"}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-xs text-green-600">
                Your information is securely stored and used only to personalize your nutrition advice.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
