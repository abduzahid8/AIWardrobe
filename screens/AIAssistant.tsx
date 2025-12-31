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
  Alert,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import React, { useState, useEffect, useRef } from "react";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Haptics from 'expo-haptics';
import Animated, { FadeInDown, FadeInUp } from "react-native-reanimated";
import { DESIGNER_STYLES, getStylePromptSuffix } from "../src/styles/designerStyles";
import StyleSelector from "../components/StyleSelector";
import { API_URL } from "../api/config";
import { colors, spacing, borderRadius, shadows } from "../src/theme";

// Message interface
interface Message {
  id: number;
  text: string;
  sender: "user" | "ai";
  timestamp: number;
}

const STYLE_STORAGE_KEY = '@selected_style';
const CHAT_HISTORY_KEY = '@ai_chat_history';
const MAX_HISTORY_MESSAGES = 50;

const AIAssistant = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const scrollViewRef = useRef<ScrollView>(null);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedStyleId, setSelectedStyleId] = useState<string | null>(null);

  // Greeting message
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning! ☀️";
    if (hour < 18) return "Good afternoon! 🌤️";
    return "Good evening! 🌙";
  };

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: `${getGreeting()} I'm your AI Stylist. Ask me anything about fashion, outfits, or style recommendations!`,
      sender: "ai",
      timestamp: Date.now(),
    },
  ]);

  // Load chat history on mount
  useEffect(() => {
    loadChatHistory();
    loadSelectedStyle();
  }, []);

  // Save chat history when messages change
  useEffect(() => {
    if (messages.length > 1) {
      saveChatHistory();
    }
  }, [messages]);

  // Auto-scroll to bottom
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [messages]);

  const loadSelectedStyle = async () => {
    try {
      const saved = await AsyncStorage.getItem(STYLE_STORAGE_KEY);
      if (saved) {
        setSelectedStyleId(saved);
      }
    } catch (error) {
      console.error('Error loading style:', error);
    }
  };

  const loadChatHistory = async () => {
    try {
      const history = await AsyncStorage.getItem(CHAT_HISTORY_KEY);
      if (history) {
        const parsed = JSON.parse(history);
        if (parsed.length > 0) {
          setMessages(parsed);
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const saveChatHistory = async () => {
    try {
      // Keep only last MAX_HISTORY_MESSAGES
      const toSave = messages.slice(-MAX_HISTORY_MESSAGES);
      await AsyncStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(toSave));
    } catch (error) {
      console.error('Error saving chat history:', error);
    }
  };

  const clearChatHistory = () => {
    Alert.alert(
      "Clear Chat History",
      "Are you sure you want to clear all messages?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear",
          style: "destructive",
          onPress: async () => {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            const greeting = {
              id: Date.now(),
              text: `${getGreeting()} I'm your AI Stylist. Ask me anything about fashion, outfits, or style recommendations!`,
              sender: "ai" as const,
              timestamp: Date.now(),
            };
            setMessages([greeting]);
            await AsyncStorage.removeItem(CHAT_HISTORY_KEY);
          },
        },
      ]
    );
  };

  const getSelectedStyleInfo = () => {
    if (!selectedStyleId) return null;
    return DESIGNER_STYLES.find(s => s.id === selectedStyleId);
  };

  // Quick action buttons
  const quickActions = [
    { icon: "sunny-outline", label: "Weather outfit", query: "What should I wear based on today's weather?" },
    { icon: "calendar-outline", label: "Date night", query: "Suggest a romantic outfit for a date tonight" },
    { icon: "briefcase-outline", label: "Work attire", query: "Professional outfit for an important meeting" },
    { icon: "fitness-outline", label: "Casual look", query: "Comfortable casual outfit for the weekend" },
  ];

  const suggestions = [
    "Suggest a casual outfit for a coffee date ☕",
    "Recommend a formal look for an interview 👔",
    "Best party outfit for tonight 🎉",
    "Summer dress ideas for a beach trip 🌴",
  ];

  const handleSend = async (textOverride?: string) => {
    const textToSend = typeof textOverride === 'string' ? textOverride : query;

    if (!textToSend.trim()) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    const userMessage: Message = {
      id: Date.now(),
      text: textToSend,
      sender: "user",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setIsLoading(true);

    try {
      const baseUrl = API_URL || "https://aiwardrobe-ivh4.onrender.com";

      // Build conversation context from recent messages
      const recentMessages = messages.slice(-6).map(m => ({
        role: m.sender === "user" ? "user" : "assistant",
        content: m.text,
      }));

      // Add style context
      const styleContext = selectedStyleId ? getStylePromptSuffix(selectedStyleId) : '';
      const enhancedQuery = styleContext
        ? `${textToSend}\n\n[User's preferred style: ${styleContext}]`
        : textToSend;

      const response = await fetch(`${baseUrl}/ai-chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: enhancedQuery,
          stylePreference: selectedStyleId || undefined,
          conversationHistory: recentMessages,
        }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.text) {
        const botMessage: Message = {
          id: Date.now() + 1,
          text: data.text,
          sender: "ai",
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error("No response text from server");
      }

    } catch (error: any) {
      console.log("AI error", error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: `Sorry, I couldn't connect to the style server. Please try again! 😔`,
        sender: "ai",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestion = (suggestion: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setQuery(suggestion);
    handleSend(suggestion);
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={styles.backButton}
          >
            <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <View style={styles.headerCenter}>
            <Text style={styles.headerTitle}>{t('aiChat.title')}</Text>
            {getSelectedStyleInfo() && (
              <Text style={styles.headerSubtitle}>
                Style: {getSelectedStyleInfo()?.name}
              </Text>
            )}
          </View>
          <TouchableOpacity onPress={clearChatHistory} style={styles.clearButton}>
            <Ionicons name="trash-outline" size={22} color={colors.text.secondary} />
          </TouchableOpacity>
        </View>

        {/* Style Selector */}
        <View style={styles.styleSelectorContainer}>
          <StyleSelector onStyleChange={loadSelectedStyle} />
        </View>

        {/* Quick Actions */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.quickActionsContainer}
          contentContainerStyle={styles.quickActionsContent}
        >
          {quickActions.map((action, index) => (
            <TouchableOpacity
              key={index}
              style={styles.quickActionButton}
              onPress={() => handleSuggestion(action.query)}
            >
              <Ionicons name={action.icon as any} size={18} color={colors.text.accent} />
              <Text style={styles.quickActionText}>{action.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Messages */}
        <KeyboardAvoidingView
          style={styles.flex}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={100}
        >
          <ScrollView
            ref={scrollViewRef}
            style={styles.messagesContainer}
            contentContainerStyle={styles.messagesContent}
            showsVerticalScrollIndicator={false}
          >
            {messages.map((message, index) => (
              <Animated.View
                key={message.id}
                entering={FadeInUp.delay(index === messages.length - 1 ? 0 : 0).duration(300)}
                style={[
                  styles.messageBubble,
                  message.sender === "user" ? styles.userBubble : styles.aiBubble,
                ]}
              >
                {message.sender === "ai" && (
                  <View style={styles.aiAvatar}>
                    <Ionicons name="sparkles" size={16} color="#FFF" />
                  </View>
                )}
                <View style={styles.messageContent}>
                  <Text style={[
                    styles.messageText,
                    message.sender === "user" ? styles.userText : styles.aiText,
                  ]}>
                    {message.text}
                  </Text>
                  <Text style={styles.messageTime}>
                    {formatTime(message.timestamp)}
                  </Text>
                </View>
              </Animated.View>
            ))}

            {isLoading && (
              <View style={[styles.messageBubble, styles.aiBubble]}>
                <View style={styles.aiAvatar}>
                  <Ionicons name="sparkles" size={16} color="#FFF" />
                </View>
                <View style={styles.typingIndicator}>
                  <View style={styles.typingDot} />
                  <View style={styles.typingDot} />
                  <View style={styles.typingDot} />
                </View>
              </View>
            )}
          </ScrollView>

          {/* Suggestions */}
          <View style={styles.suggestionsContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {suggestions.map((sugg, index) => (
                <TouchableOpacity
                  onPress={() => handleSuggestion(sugg)}
                  key={index}
                  style={styles.suggestionChip}
                >
                  <Text style={styles.suggestionText}>{sugg}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* Input */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={query}
              onChangeText={setQuery}
              placeholder={t('aiChat.placeholder')}
              placeholderTextColor={colors.text.muted}
              multiline
              maxLength={500}
            />
            <TouchableOpacity
              onPress={() => handleSend()}
              disabled={isLoading || !query.trim()}
              style={[
                styles.sendButton,
                (!query.trim() || isLoading) && styles.sendButtonDisabled,
              ]}
            >
              <Ionicons
                name="send"
                size={20}
                color={query.trim() && !isLoading ? "#FFF" : colors.text.muted}
              />
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

export default AIAssistant;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  safeArea: {
    flex: 1,
  },
  flex: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.m,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.surface,
  },
  backButton: {
    padding: spacing.xs,
  },
  headerCenter: {
    flex: 1,
    marginLeft: spacing.m,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: colors.text.primary,
  },
  headerSubtitle: {
    fontSize: 12,
    color: colors.text.accent,
    marginTop: 2,
  },
  clearButton: {
    padding: spacing.xs,
  },
  styleSelectorContainer: {
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.s,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  quickActionsContainer: {
    maxHeight: 50,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  quickActionsContent: {
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.s,
    gap: spacing.s,
  },
  quickActionButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surfaceHighlight,
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    marginRight: spacing.s,
    gap: spacing.xs,
  },
  quickActionText: {
    fontSize: 13,
    fontWeight: "500",
    color: colors.text.primary,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: spacing.m,
    paddingBottom: spacing.xl,
  },
  messageBubble: {
    flexDirection: "row",
    marginBottom: spacing.m,
    maxWidth: "85%",
  },
  userBubble: {
    alignSelf: "flex-end",
    flexDirection: "row-reverse",
  },
  aiBubble: {
    alignSelf: "flex-start",
  },
  aiAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.text.accent,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.s,
  },
  messageContent: {
    backgroundColor: colors.surface,
    padding: spacing.m,
    borderRadius: borderRadius.l,
    ...shadows.soft,
  },
  messageText: {
    fontSize: 15,
    lineHeight: 22,
  },
  userText: {
    color: colors.text.primary,
  },
  aiText: {
    color: colors.text.primary,
  },
  messageTime: {
    fontSize: 10,
    color: colors.text.muted,
    marginTop: spacing.xs,
    textAlign: "right",
  },
  typingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    padding: spacing.m,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.l,
    gap: 4,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.text.muted,
  },
  suggestionsContainer: {
    paddingVertical: spacing.s,
    paddingHorizontal: spacing.m,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: colors.surface,
  },
  suggestionChip: {
    backgroundColor: colors.surfaceHighlight,
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.s,
    borderRadius: borderRadius.full,
    marginRight: spacing.s,
  },
  suggestionText: {
    fontSize: 13,
    color: colors.text.primary,
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "flex-end",
    padding: spacing.m,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    gap: spacing.s,
  },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 100,
    backgroundColor: colors.surfaceHighlight,
    borderRadius: borderRadius.l,
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.s,
    fontSize: 15,
    color: colors.text.primary,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.text.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  sendButtonDisabled: {
    backgroundColor: colors.surfaceHighlight,
  },
});