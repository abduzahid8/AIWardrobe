/**
 * Push Notification Service
 * Handles notification scheduling and engagement for AIWardrobe
 */

import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { createLogger } from '../utils/logger';

const logger = createLogger('Notifications');

const NOTIFICATION_STORAGE_KEY = 'notificationSettings';

// ============================================
// CONFIGURATION
// ============================================

// Set notification handler
Notifications.setNotificationHandler({
    handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: true,
        shouldShowBanner: true,
        shouldShowList: true,
    }),
});

// Notification types
export type NotificationType =
    | 'daily_outfit'
    | 'event_reminder'
    | 'weather_alert'
    | 'unused_items'
    | 'style_tip'
    | 'sale_alert';

// Default notification settings
export interface NotificationSettings {
    enabled: boolean;
    dailyOutfit: {
        enabled: boolean;
        time: { hour: number; minute: number }; // e.g., { hour: 7, minute: 0 } for 7 AM
    };
    eventReminders: boolean;
    weatherAlerts: boolean;
    unusedItems: boolean;
    styleTips: boolean;
    saleAlerts: boolean;
}

const DEFAULT_SETTINGS: NotificationSettings = {
    enabled: true,
    dailyOutfit: {
        enabled: true,
        time: { hour: 7, minute: 0 },
    },
    eventReminders: true,
    weatherAlerts: true,
    unusedItems: true,
    styleTips: false,
    saleAlerts: false,
};

// ============================================
// NOTIFICATION CONTENT
// ============================================

const MORNING_OUTFIT_MESSAGES = [
    { title: "Good morning! ☀️", body: "What's your plan today? Let me suggest the perfect outfit!" },
    { title: "Time to get dressed! 👔", body: "I've got some great outfit ideas based on today's weather." },
    { title: "Rise and shine! ✨", body: "Check out today's outfit recommendation!" },
    { title: "Ready to slay? 💪", body: "Here's your personalized outfit for today." },
];

const WEATHER_MESSAGES = {
    rain: { title: "☔ Rain Alert!", body: "It's going to rain today. Don't forget your umbrella and waterproof shoes!" },
    cold: { title: "🥶 Bundle Up!", body: "It's chilly today. Time for layers and a warm jacket!" },
    hot: { title: "☀️ It's Hot!", body: "Stay cool in breathable fabrics today." },
    clear: { title: "Perfect Weather! 🌤", body: "Great day for your favorite outfit!" },
};

const UNUSED_ITEM_MESSAGES = [
    { title: "Forgotten Item 👀", body: "Your {item} hasn't been worn in {days} days. Give it some love!" },
    { title: "Wardrobe Reminder 📦", body: "Don't forget about your {item}! It's been waiting for {days} days." },
];

const EVENING_WEARLOG_MESSAGES = [
    { title: "What did you wear today? 👗", body: "Log your outfit to keep your streak alive and improve tomorrow's suggestions." },
    { title: "Quick style log 📝", body: "One tap to log today's outfit. Your streak is counting on you!" },
    { title: "End of day check-in ✨", body: "How was today's outfit? Log it now to unlock better suggestions." },
];

const WEEKLY_INSIGHTS_MESSAGES = [
    { title: "Your week in style 📊", body: "See your closet utilization, top colors, and style patterns!" },
    { title: "Weekly style report 🏆", body: "Check out your wardrobe stats and discover forgotten items." },
];

const STYLE_TIPS = [
    { title: "Style Tip 💡", body: "Try tucking your shirt in for a more polished look!" },
    { title: "Color Combo 🎨", body: "Navy and white is always a winning combination." },
    { title: "Outfit Hack ✨", body: "Roll your sleeves for an instant casual upgrade." },
    { title: "Pro Tip 👔", body: "Match your belt with your shoes for a cohesive look." },
];

// ============================================
// NOTIFICATION SERVICE
// ============================================

class NotificationService {
    private settings: NotificationSettings = DEFAULT_SETTINGS;
    private expoPushToken: string | null = null;

    async initialize() {
        try {
            // Load saved settings
            const savedSettings = await AsyncStorage.getItem(NOTIFICATION_STORAGE_KEY);
            if (savedSettings) {
                this.settings = { ...DEFAULT_SETTINGS, ...JSON.parse(savedSettings) };
            }

            // Request permissions
            await this.requestPermissions();

            // Schedule recurring notifications
            if (this.settings.enabled) {
                await this.scheduleRecurringNotifications();
            }

            logger.info('📱 Notification service initialized');
        } catch (error) {
            console.error('Failed to initialize notifications:', error);
        }
    }

    async requestPermissions(): Promise<boolean> {
        if (!Device.isDevice) {
            logger.warn('Notifications only work on physical devices');
            return false;
        }

        const { status: existingStatus } = await Notifications.getPermissionsAsync();
        let finalStatus = existingStatus;

        if (existingStatus !== 'granted') {
            const { status } = await Notifications.requestPermissionsAsync();
            finalStatus = status;
        }

        if (finalStatus !== 'granted') {
            logger.warn('Notification permission denied');
            return false;
        }

        // Get push token for server notifications
        try {
            const token = await Notifications.getExpoPushTokenAsync({
                projectId: process.env.EXPO_PUBLIC_PROJECT_ID,
            });
            this.expoPushToken = token.data;
            logger.info('📱 Push token', this.expoPushToken);
        } catch (error) {
            console.error('Failed to get push token:', error);
        }

        // iOS specific setup
        if (Platform.OS === 'ios') {
            await Notifications.setNotificationCategoryAsync('outfit', [
                {
                    identifier: 'view',
                    buttonTitle: 'View Outfit',
                    options: { opensAppToForeground: true },
                },
                {
                    identifier: 'snooze',
                    buttonTitle: 'Remind Later',
                    options: { opensAppToForeground: false },
                },
            ]);
        }

        return true;
    }

    async getSettings(): Promise<NotificationSettings> {
        return this.settings;
    }

    async updateSettings(newSettings: Partial<NotificationSettings>): Promise<void> {
        this.settings = { ...this.settings, ...newSettings };
        await AsyncStorage.setItem(NOTIFICATION_STORAGE_KEY, JSON.stringify(this.settings));

        // Reschedule notifications
        await Notifications.cancelAllScheduledNotificationsAsync();
        if (this.settings.enabled) {
            await this.scheduleRecurringNotifications();
        }
    }

    async scheduleRecurringNotifications() {
        // Cancel existing scheduled notifications
        await Notifications.cancelAllScheduledNotificationsAsync();

        // Schedule daily outfit notification (morning)
        if (this.settings.dailyOutfit.enabled) {
            await this.scheduleDailyOutfitNotification();
        }

        // Schedule evening wear-log reminder
        if (this.settings.dailyOutfit.enabled) {
            await this.scheduleEveningWearLogNotification();
        }

        // Schedule weekly insights (Sunday)
        await this.scheduleWeeklyInsightsNotification();

        // Schedule weekly unused items reminder
        if (this.settings.unusedItems) {
            await this.scheduleWeeklyUnusedItemsReminder();
        }

        // Schedule weekly style tip
        if (this.settings.styleTips) {
            await this.scheduleWeeklyStyleTip();
        }
    }

    /**
     * Schedule daily morning outfit notification
     */
    async scheduleDailyOutfitNotification() {
        const { hour, minute } = this.settings.dailyOutfit.time;
        const randomMessage = MORNING_OUTFIT_MESSAGES[
            Math.floor(Math.random() * MORNING_OUTFIT_MESSAGES.length)
        ];

        await Notifications.scheduleNotificationAsync({
            content: {
                title: randomMessage.title,
                body: randomMessage.body,
                data: { type: 'daily_outfit', screen: 'Main' },
                categoryIdentifier: 'outfit',
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.DAILY,
                hour,
                minute,
            },
        });

        logger.info(`📱 Daily outfit notification scheduled for ${hour}:${minute.toString().padStart(2, '0')}`);
    }

    /**
     * Schedule evening wear-log reminder (8 PM daily)
     */
    async scheduleEveningWearLogNotification() {
        const randomMessage = EVENING_WEARLOG_MESSAGES[
            Math.floor(Math.random() * EVENING_WEARLOG_MESSAGES.length)
        ];

        await Notifications.scheduleNotificationAsync({
            content: {
                title: randomMessage.title,
                body: randomMessage.body,
                data: { type: 'evening_wearlog', screen: 'MyCloset' },
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.DAILY,
                hour: 20,
                minute: 0,
            },
        });

        logger.info('📱 Evening wear-log notification scheduled for 20:00');
    }

    /**
     * Schedule weekly insights notification (Sunday 10 AM)
     */
    async scheduleWeeklyInsightsNotification() {
        const randomMessage = WEEKLY_INSIGHTS_MESSAGES[
            Math.floor(Math.random() * WEEKLY_INSIGHTS_MESSAGES.length)
        ];

        await Notifications.scheduleNotificationAsync({
            content: {
                title: randomMessage.title,
                body: randomMessage.body,
                data: { type: 'weekly_insights', screen: 'WardrobeAnalytics' },
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.WEEKLY,
                weekday: 1, // Sunday
                hour: 10,
                minute: 0,
            },
        });

        logger.info('📱 Weekly insights notification scheduled for Sunday 10:00');
    }

    /**
     * Schedule notification for upcoming calendar event
     */
    async scheduleEventReminder(event: {
        title: string;
        date: Date;
        occasion?: string;
    }) {
        if (!this.settings.eventReminders) return;

        // Schedule for evening before the event
        const reminderDate = new Date(event.date);
        reminderDate.setDate(reminderDate.getDate() - 1);
        reminderDate.setHours(20, 0, 0, 0);

        if (reminderDate < new Date()) return; // Skip if in the past

        await Notifications.scheduleNotificationAsync({
            content: {
                title: `Outfit for tomorrow 👔`,
                body: `You have "${event.title}" tomorrow. Pick your outfit now!`,
                data: { type: 'event_reminder', event: event.title, screen: 'OutfitAI' },
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.DATE,
                date: reminderDate,
            },
        });

        logger.info(`📱 Event reminder scheduled for ${event.title}`);
    }

    /**
     * Send weather-based outfit alert
     */
    async sendWeatherAlert(condition: 'rain' | 'cold' | 'hot' | 'clear') {
        if (!this.settings.weatherAlerts) return;

        const message = WEATHER_MESSAGES[condition];

        await Notifications.scheduleNotificationAsync({
            content: {
                title: message.title,
                body: message.body,
                data: { type: 'weather_alert', condition, screen: 'Main' },
            },
            trigger: null, // Send immediately
        });
    }

    /**
     * Schedule weekly unused items reminder
     */
    async scheduleWeeklyUnusedItemsReminder() {
        await Notifications.scheduleNotificationAsync({
            content: {
                title: "📦 Wardrobe Check",
                body: "Some items haven't been worn lately. Give them a chance!",
                data: { type: 'unused_items', screen: 'WardrobeAnalytics' },
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.WEEKLY,
                weekday: 2, // Monday (1=Sunday, 2=Monday...)
                hour: 10,
                minute: 0,
            },
        });
    }

    /**
     * Schedule weekly style tip
     */
    async scheduleWeeklyStyleTip() {
        const randomTip = STYLE_TIPS[Math.floor(Math.random() * STYLE_TIPS.length)];

        await Notifications.scheduleNotificationAsync({
            content: {
                title: randomTip.title,
                body: randomTip.body,
                data: { type: 'style_tip' },
            },
            trigger: {
                type: Notifications.SchedulableTriggerInputTypes.WEEKLY,
                weekday: 6, // Friday (1=Sunday, 6=Friday)
                hour: 12,
                minute: 0,
            },
        });
    }

    /**
     * Send notification for specific unused item
     */
    async notifyUnusedItem(item: { type: string; lastWorn: Date }) {
        if (!this.settings.unusedItems) return;

        const daysSinceWorn = Math.floor(
            (Date.now() - item.lastWorn.getTime()) / (1000 * 60 * 60 * 24)
        );

        if (daysSinceWorn < 30) return; // Only notify if not worn in 30+ days

        const template = UNUSED_ITEM_MESSAGES[
            Math.floor(Math.random() * UNUSED_ITEM_MESSAGES.length)
        ];

        const title = template.title;
        const body = template.body
            .replace('{item}', item.type)
            .replace('{days}', daysSinceWorn.toString());

        await Notifications.scheduleNotificationAsync({
            content: {
                title,
                body,
                data: { type: 'unused_items', itemType: item.type },
            },
            trigger: null,
        });
    }

    /**
     * Send sale alert
     */
    async sendSaleAlert(brand: string, discount: number) {
        if (!this.settings.saleAlerts) return;

        await Notifications.scheduleNotificationAsync({
            content: {
                title: `🏷️ Sale Alert!`,
                body: `${brand} is ${discount}% off! Shop your favorites now.`,
                data: { type: 'sale_alert', brand, screen: 'Shopping' },
            },
            trigger: null,
        });
    }

    /**
     * Get push token for server
     */
    getPushToken(): string | null {
        return this.expoPushToken;
    }

    /**
     * Cancel all notifications
     */
    async cancelAll() {
        await Notifications.cancelAllScheduledNotificationsAsync();
    }

    /**
     * Get scheduled notifications
     */
    async getScheduledNotifications() {
        return Notifications.getAllScheduledNotificationsAsync();
    }
}

// Add notification listeners
export const addNotificationListeners = (
    onNotificationReceived?: (notification: Notifications.Notification) => void,
    onNotificationPressed?: (response: Notifications.NotificationResponse) => void
) => {
    const receivedListener = Notifications.addNotificationReceivedListener((notification: Notifications.Notification) => {
        logger.debug('📩 Notification received', notification);
        onNotificationReceived?.(notification);
    });

    const responseListener = Notifications.addNotificationResponseReceivedListener((response: Notifications.NotificationResponse) => {
        logger.debug('📱 Notification pressed', response);
        onNotificationPressed?.(response);

        // Handle navigation based on notification data
        const data = response.notification.request.content.data;
        if (data?.screen) {
            // Navigation handled by RootNavigator's notification listener
            logger.debug(`Navigate to: ${data.screen}`);
        }
    });

    return () => {
        receivedListener.remove();
        responseListener.remove();
    };
};

// Export singleton instance
export const notificationService = new NotificationService();
export default notificationService;
