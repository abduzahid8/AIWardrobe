import {
  ActivityIndicator,
  Linking,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useState } from "react";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
// @ts-ignore - игнорируем ошибку, если путь немного отличается
import { API_URL } from "../api/config";

// Определяем типы для сообщений, чтобы TypeScript не ругался
interface Message {
  id: number;
  text: string;
  sender: "user" | "ai";
}

const AIAssistant = () => {
  const navigation = useNavigation();
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Указываем, что это массив сообщений
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I am your AI Stylist. Ask me about outfits! 👗",
      sender: "ai",
    },
  ]);

  const suggestions = [
    "Suggest a casual outfit for a coffee date ☕",
    "Recommend a formal look for an interview 👔",
    "Best party outfit for tonight 🎉",
    "Summer dress ideas for a beach trip 🌴",
  ];

  // Функция отправки
  const handleSend = async (textOverride?: string) => {
    const textToSend = typeof textOverride === 'string' ? textOverride : query;

    if (!textToSend.trim()) return;

    const userMessage: Message = { id: Date.now(), text: textToSend, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setIsLoading(true);

    try {
      // Используем API_URL из конфига.
      const baseUrl = API_URL || "https://aiwardrobe-ivh4.onrender.com";

      const response = await fetch(`${baseUrl}/ai-chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: textToSend,
        }),
      }
      );

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.text) {
        const aiResponse = data.text;

        const enhancedResponse = aiResponse
          .replace("http", " [Link](")
          .replace(" ", ") ");

        const botMessage: Message = {
          id: Date.now() + 1,
          text: enhancedResponse,
          sender: "ai"
        };

        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error("No response text from server");
      }

    } catch (error: any) {
      console.log("AI error", error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: `Sorry, server connection error. Try again! 😔`,
        sender: "ai",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Типизируем аргумент как string
  const handleSuggestion = (suggestion: string) => {
    setQuery(suggestion);
    handleSend(suggestion);
  };

  return (
    <SafeAreaView className="flex-1 bg-gray-100">
      <View className="flex-row justify-between items-center p-4 bg-white border-b border-gray-200">
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text className="text-xl font-bold text-gray-800">
          AI Fashion Assistant
        </Text>
        <View className="w-6" />
      </View>

      <ScrollView
        className="flex-1 p-4"
        contentContainerStyle={{ paddingBottom: 20 }}
      >
        {messages.map((message) => (
          <View
            key={message.id}
            className={`mb-4 p-3 rounded-lg max-w-[80%] ${message.sender == "user"
                ? "bg-cyan-200 self-end"
                : "bg-cyan-100 self-start"
              }`}
          >
            <Text className="text-base text-gray-800">{message.text}</Text>
            {message.sender === "ai" &&
              message.text.includes("[Link]") &&
              message.text
                .split("[Link](")
                .slice(1)
                .map((part, index) => {
                  const [url, rest] = part.split(") ");
                  if (url) {
                    return (
                      <TouchableOpacity
                        key={index}
                        onPress={() => Linking.openURL(url)}
                        className="mt-2"
                      >
                        <Text className="text-blue-600 text-sm">
                          🌐 Visit {url}
                        </Text>
                      </TouchableOpacity>
                    );
                  }
                  return null;
                })}
          </View>
        ))}

        {isLoading && (
          <View className="flex items-center mt-4">
            <ActivityIndicator size={"large"} color="#1e90ff" />
            <Text className="text-gray-600 mt-2">Styling...</Text>
          </View>
        )}
      </ScrollView>

      <View className="p-4 bg-white border-t border-gray-200">
        <Text className="text-lg font-bold text-gray-800 mb-2">
          Quick Suggestions:
        </Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {suggestions.map((sugg, index) => (
            <TouchableOpacity
              onPress={() => handleSuggestion(sugg)}
              key={index}
              className="bg-gray-200 px-4 py-2 rounded-full mr-2"
            >
              <Text>{sugg}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <View className="flex-row items-center p-4 bg-white border-t border-gray-200">
        <TextInput
          className="flex-1 h-10 bg-gray-100 rounded-full px-4 text-base text-gray-800"
          value={query}
          onChangeText={setQuery}
          placeholder="Ask me anything about fashion..."
          placeholderTextColor={"#999"}
        />
        <TouchableOpacity
          onPress={() => handleSend()}
          disabled={isLoading}
          className={`ml-2 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center`}
        >
          <Ionicons name="send" size={20} color={isLoading ? "#ccc" : "#fff"} />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

export default AIAssistant;

const styles = StyleSheet.create({});