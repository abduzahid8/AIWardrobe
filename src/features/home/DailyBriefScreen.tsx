import React, { useState, useEffect } from "react";
import {
    View,
    Text,
    Image,
    ScrollView,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ActivityIndicator,
    Platform,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import moment from "moment";
import { colors, shadows, spacing, typography } from "../../theme";
import { useNavigation } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import * as Location from 'expo-location';
import axios from "axios";

const { width, height } = Dimensions.get("window");
const WEATHER_API_KEY = "acec1d31ef3e181c0ca471ac4db642ff";

const DailyBriefScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [weather, setWeather] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        (async () => {
            try {
                let { status } = await Location.requestForegroundPermissionsAsync();
                if (status !== 'granted') {
                    setLoading(false);
                    return;
                }

                let location = await Location.getCurrentPositionAsync({});
                const { latitude, longitude } = location.coords;

                const response = await axios.get(
                    `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${WEATHER_API_KEY}`
                );

                setWeather({
                    temp: Math.round(response.data.main.temp),
                    description: response.data.weather[0].description,
                    city: response.data.name
                });
            } catch (e) {
                console.log(e);
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <Text style={styles.date}>{moment().format("dddd, MMM Do")}</Text>
                    <TouchableOpacity onPress={() => (navigation as any).navigate("Profile")}>
                        <View style={styles.profileButton}>
                            <Ionicons name="person-outline" size={20} color={colors.text.primary} />
                        </View>
                    </TouchableOpacity>
                </View>

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>

                    {/* Hero Section: The "Magazine Cover" */}
                    <View style={styles.heroContainer}>
                        <Text style={styles.heroTitle}>{t('home.dailyBrief')}</Text>

                        <View style={styles.weatherContainer}>
                            {loading ? (
                                <ActivityIndicator color={colors.text.secondary} />
                            ) : (
                                <Text style={styles.weatherText}>
                                    {weather ? `${weather.temp}° ${weather.city}` : "24° Tashkent"} • {weather ? weather.description : "Sunny"}
                                </Text>
                            )}
                        </View>

                        {/* Outfit of the Day Card */}
                        <View style={[styles.outfitCard, shadows.medium]}>
                            <Image
                                source={{ uri: "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?q=80&w=1000&auto=format&fit=crop" }}
                                style={styles.outfitImage}
                                resizeMode="cover"
                            />
                            <View style={styles.outfitOverlay}>
                                <Text style={styles.outfitLabel}>{t('home.todaysLook')}</Text>
                                <Text style={styles.outfitTitle}>Casual Chic for Work</Text>
                            </View>
                        </View>
                    </View>

                    {/* Stylist Insight */}
                    <View style={styles.section}>
                        <Text style={styles.sectionTitle}>{t('home.stylistInsight')}</Text>
                        <View style={styles.insightCard}>
                            <Text style={styles.insightText}>
                                "It's a bit chilly this morning. Layer that beige trench coat over your white tee for an effortless, polished look."
                            </Text>
                            <View style={styles.stylistSignature}>
                                <View style={styles.stylistAvatar} />
                                <Text style={styles.stylistName}>AI Stylist</Text>
                            </View>
                        </View>
                    </View>

                </ScrollView>

                {/* Floating Action Button (FAB) */}
                <TouchableOpacity
                    style={[styles.fab, shadows.medium]}
                    onPress={() => (navigation as any).navigate("AIChat")}
                >
                    <Ionicons name="sparkles" size={24} color="#FFF" />
                    <Text style={styles.fabText}>{t('home.askStylist')}</Text>
                </TouchableOpacity>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingHorizontal: spacing.l,
        paddingVertical: spacing.m,
    },
    date: {
        fontSize: 14,
        color: colors.text.secondary,
        textTransform: "uppercase",
        letterSpacing: 1,
        fontWeight: "600",
    },
    profileButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: colors.border,
        alignItems: "center",
        justifyContent: "center",
    },
    scrollContent: {
        paddingBottom: 100,
    },
    heroContainer: {
        paddingHorizontal: spacing.l,
        marginBottom: spacing.xl,
    },
    heroTitle: {
        fontSize: 48,
        color: colors.text.primary,
        fontFamily: Platform.OS === 'ios' ? 'Didot' : 'serif', // Editorial Font
        fontWeight: "400",
        marginBottom: spacing.xs,
    },
    weatherContainer: {
        marginBottom: spacing.l,
    },
    weatherText: {
        fontSize: 16,
        color: colors.text.secondary,
        fontStyle: "italic",
    },
    outfitCard: {
        width: "100%",
        height: height * 0.55,
        borderRadius: 2, // Sharp corners for editorial look
        overflow: "hidden",
        backgroundColor: colors.surfaceHighlight,
        position: "relative",
    },
    outfitImage: {
        width: "100%",
        height: "100%",
    },
    outfitOverlay: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        padding: spacing.l,
        backgroundColor: "rgba(0,0,0,0.2)",
    },
    outfitLabel: {
        color: "#FFF",
        fontSize: 12,
        textTransform: "uppercase",
        letterSpacing: 1,
        marginBottom: spacing.xs,
    },
    outfitTitle: {
        color: "#FFF",
        fontSize: 24,
        fontFamily: Platform.OS === 'ios' ? 'Didot' : 'serif',
    },
    section: {
        paddingHorizontal: spacing.l,
        marginBottom: spacing.xl,
    },
    sectionTitle: {
        fontSize: 18,
        color: colors.text.primary,
        marginBottom: spacing.m,
        fontWeight: "600",
    },
    insightCard: {
        backgroundColor: colors.surface,
        padding: spacing.l,
        borderLeftWidth: 2,
        borderLeftColor: colors.text.accent,
    },
    insightText: {
        fontSize: 18,
        color: colors.text.primary,
        lineHeight: 28,
        fontFamily: Platform.OS === 'ios' ? 'Georgia' : 'serif',
        fontStyle: "italic",
        marginBottom: spacing.m,
    },
    stylistSignature: {
        flexDirection: "row",
        alignItems: "center",
    },
    stylistAvatar: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: colors.text.accent,
        marginRight: spacing.s,
    },
    stylistName: {
        fontSize: 14,
        color: colors.text.secondary,
        fontWeight: "600",
    },
    fab: {
        position: "absolute",
        bottom: 32,
        alignSelf: "center",
        backgroundColor: colors.text.primary,
        paddingVertical: 16,
        paddingHorizontal: 32,
        borderRadius: 100,
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    fabText: {
        color: "#FFF",
        fontSize: 16,
        fontWeight: "600",
    },
});

export default DailyBriefScreen;
