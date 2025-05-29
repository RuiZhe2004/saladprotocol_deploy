"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Salad, Camera, Send, User, Bot, LogOut, Loader2 } from "lucide-react"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  foodAnalysis?: any
}

interface FoodAnalysis {
  food_items: Array<{
    name: string
    calories: number
    protein: number
    carbs: number
    fat: number
    portion_size: string
  }>
  total_calories: number
  total_protein: number
  total_carbs: number
  total_fat: number
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [username, setUsername] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [lastFoodAnalysis, setLastFoodAnalysis] = useState<FoodAnalysis | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  useEffect(() => {
    const storedUsername = localStorage.getItem("username")
    if (!storedUsername) {
      router.push("/")
      return
    }
    setUsername(storedUsername)

    // Add welcome message
    setMessages([
      {
        id: "1",
        role: "assistant",
        content: `Hello ${storedUsername}! I'm your AI nutritionist. I can help you with nutrition advice, meal planning, and analyze your food photos. How can I assist you today?`,
        timestamp: new Date(),
      },
    ])
  }, [router])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleLogout = () => {
    localStorage.removeItem("username")
    router.push("/")
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file)
    }
  }

  const analyzeFoodImage = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    const formData = new FormData()
    formData.append("image", selectedFile)
    formData.append("username", username)

    try {
      const response = await fetch("/api/food/analyze", {
        method: "POST",
        body: formData,
      })

      const analysis = await response.json()

      if (response.ok) {
        setLastFoodAnalysis(analysis)

        // Add food analysis message
        const analysisMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: `I've analyzed your food image! Here's what I found:\n\n${analysis.food_items
            .map(
              (item: any) =>
                `🍽️ **${item.name}** (${item.portion_size})\n   Calories: ${item.calories} | Protein: ${item.protein}g | Carbs: ${item.carbs}g | Fat: ${item.fat}g`,
            )
            .join(
              "\n\n",
            )}\n\n**Total:** ${analysis.total_calories} calories, ${analysis.total_protein}g protein, ${analysis.total_carbs}g carbs, ${analysis.total_fat}g fat\n\nFeel free to ask me any questions about this meal!`,
          timestamp: new Date(),
          foodAnalysis: analysis,
        }

        setMessages((prev) => [...prev, analysisMessage])
        setSelectedFile(null)
        if (fileInputRef.current) {
          fileInputRef.current.value = ""
        }
      } else {
        alert("Failed to analyze image. Please try again.")
      }
    } catch (error) {
      console.error("Food analysis error:", error)
      alert("An error occurred while analyzing the image.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input.trim(),
          username,
          lastFoodAnalysis,
          conversationHistory: messages.slice(-10), // Send last 10 messages for context
        }),
      })

      const data = await response.json()

      if (response.ok) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.response,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, assistantMessage])
      } else {
        throw new Error("Failed to get response")
      }
    } catch (error) {
      console.error("Chat error:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100">
      {/* Header */}
      <div className="bg-white border-b border-green-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-green-500 p-2 rounded-full">
              <Salad className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-green-800">Salad Protocol</h1>
              <p className="text-sm text-green-600">AI Nutritionist</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="border-green-300 text-green-700">
              {username}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={handleLogout}
              className="border-green-300 text-green-700 hover:bg-green-50"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="max-w-4xl mx-auto p-4">
        <Card className="h-[calc(100vh-200px)] flex flex-col shadow-lg border-green-200">
          {/* Messages */}
          <CardContent className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`flex items-start gap-3 max-w-[80%] ${message.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div className={`p-2 rounded-full ${message.role === "user" ? "bg-green-500" : "bg-emerald-500"}`}>
                    {message.role === "user" ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>
                  <div
                    className={`p-4 rounded-lg ${
                      message.role === "user" ? "bg-green-500 text-white" : "bg-white border border-green-200"
                    }`}
                  >
                    <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                    {message.foodAnalysis && (
                      <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200">
                        <div className="text-xs text-green-600 font-medium mb-2">Food Analysis Data</div>
                        <div className="text-xs text-green-700">
                          Total: {message.foodAnalysis.total_calories} cal | P: {message.foodAnalysis.total_protein}g |
                          C: {message.foodAnalysis.total_carbs}g | F: {message.foodAnalysis.total_fat}g
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-full bg-emerald-500">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                  <div className="p-4 rounded-lg bg-white border border-green-200">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-green-500" />
                      <span className="text-sm text-green-600">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>

          {/* Food Image Upload */}
          {selectedFile && (
            <div className="px-6 py-3 border-t border-green-200 bg-green-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Camera className="h-5 w-5 text-green-600" />
                  <span className="text-sm text-green-700">{selectedFile.name}</span>
                </div>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setSelectedFile(null)}
                    className="border-green-300 text-green-700"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={analyzeFoodImage}
                    disabled={isAnalyzing}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      "Analyze Food"
                    )}
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="p-6 border-t border-green-200 bg-white">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                className="border-green-300 text-green-700 hover:bg-green-50"
              >
                <Camera className="h-4 w-4" />
              </Button>
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about nutrition, upload a food photo, or get meal advice..."
                className="flex-1 border-green-300 focus:border-green-500 focus:ring-green-500"
                disabled={isLoading}
              />
              <Button type="submit" disabled={isLoading || !input.trim()} className="bg-green-600 hover:bg-green-700">
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </Card>
      </div>
    </div>
  )
}
