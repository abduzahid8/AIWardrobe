import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    FlatList,
    ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';

const { width } = Dimensions.get('window');

// Brand List
const BRANDS = [
    { id: '1', name: 'A.P.C.' },
    { id: '2', name: 'AMI Paris' },
    { id: '3', name: 'ASOS' },
    { id: '4', name: 'Acne Studios' },
    { id: '5', name: 'Adidas' },
    { id: '6', name: 'Aime Leon Dore' },
    { id: '7', name: 'Alexander McQueen' },
    { id: '8', name: 'Balenciaga' },
    { id: '9', name: 'Burberry' },
    { id: '10', name: 'Calvin Klein' },
    { id: '11', name: 'Carhartt WIP' },
    { id: '12', name: 'Celine' },
    { id: '13', name: 'Cos' },
    { id: '14', name: 'Dior' },
    { id: '15', name: 'Fear of God' },
    { id: '16', name: 'Gucci' },
    { id: '17', name: 'H&M' },
    { id: '18', name: 'Hugo Boss' },
    { id: '19', name: 'Jacquemus' },
    { id: '20', name: 'Kenzo' },
    { id: '21', name: 'Lacoste' },
    { id: '22', name: 'Loewe' },
    { id: '23', name: 'Louis Vuitton' },
    { id: '24', name: 'Massimo Dutti' },
    { id: '25', name: 'Nike' },
    { id: '26', name: 'Off-White' },
    { id: '27', name: 'Polo Ralph Lauren' },
    { id: '28', name: 'Prada' },
    { id: '29', name: 'Stussy' },
    { id: '30', name: 'Supreme' },
    { id: '31', name: 'The North Face' },
    { id: '32', name: 'Tommy Hilfiger' },
    { id: '33', name: 'Uniqlo' },
    { id: '34', name: 'Versace' },
    { id: '35', name: 'Zara' },
];

// Progress Bar
const ProgressBar = ({ steps, currentStep }: { steps: number; currentStep: number }) => (
    <View style={styles.progressContainer}>
        {Array.from({ length: steps }).map((_, index) => (
            <View
                key={index}
                style={[
                    styles.progressStep,
                    index <= currentStep ? styles.progressStepActive : styles.progressStepInactive,
                ]}
            />
        ))}
    </View>
);

// Brand Row
const BrandRow = ({
    brand,
    isLiked,
    onToggle
}: {
    brand: { id: string; name: string };
    isLiked: boolean;
    onToggle: () => void;
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <TouchableOpacity
            style={styles.brandRow}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                scale.value = withSpring(1.05, {}, () => {
                    scale.value = withSpring(1);
                });
                onToggle();
            }}
            activeOpacity={0.8}
        >
            <Text style={styles.brandName}>{brand.name}</Text>
            <Animated.View style={animatedStyle}>
                <Ionicons
                    name={isLiked ? 'heart' : 'heart-outline'}
                    size={22}
                    color={isLiked ? '#000' : '#9CA3AF'}
                />
            </Animated.View>
        </TouchableOpacity>
    );
};

const BrandSelectionScreen = () => {
    const navigation = useNavigation();
    const [searchQuery, setSearchQuery] = useState('');
    const [likedBrands, setLikedBrands] = useState<string[]>([]);

    const filteredBrands = BRANDS.filter(brand =>
        brand.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const toggleBrand = (brandId: string) => {
        setLikedBrands(prev =>
            prev.includes(brandId)
                ? prev.filter(id => id !== brandId)
                : [...prev, brandId]
        );
    };

    const canContinue = likedBrands.length >= 3;

    const handleContinue = () => {
        if (canContinue) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            // Navigate to next onboarding step or main app
            (navigation as any).navigate('Main');
        }
    };

    const handleSkip = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        (navigation as any).navigate('Main');
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header with Progress & Skip */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={styles.header}
                >
                    <ProgressBar steps={5} currentStep={4} />
                    <TouchableOpacity onPress={handleSkip} style={styles.skipButton}>
                        <Text style={styles.skipText}>Skip</Text>
                    </TouchableOpacity>
                </Animated.View>

                {/* Icon */}
                <Animated.View
                    entering={FadeInUp.delay(100).springify()}
                    style={styles.iconContainer}
                >
                    <View style={styles.tagIcon}>
                        <Ionicons name="pricetag-outline" size={24} color="#000" />
                    </View>
                </Animated.View>

                {/* Title */}
                <Animated.View
                    entering={FadeInUp.delay(150).springify()}
                    style={styles.titleContainer}
                >
                    <Text style={styles.title}>
                        Choose 3 or{'\n'}more <Text style={styles.titleItalic}>brands</Text>
                    </Text>
                    <Text style={styles.subtitle}>
                        Choose brands of clothes you currently own or want
                    </Text>
                </Animated.View>

                {/* Search Input */}
                <Animated.View
                    entering={FadeInUp.delay(200).springify()}
                    style={styles.searchContainer}
                >
                    <TextInput
                        style={styles.searchInput}
                        placeholder="Search or add brands..."
                        placeholderTextColor="#9CA3AF"
                        value={searchQuery}
                        onChangeText={setSearchQuery}
                    />
                    <Ionicons name="search" size={20} color="#9CA3AF" style={styles.searchIcon} />
                </Animated.View>

                {/* Brand List */}
                <FlatList
                    data={filteredBrands}
                    keyExtractor={(item) => item.id}
                    renderItem={({ item, index }) => (
                        <Animated.View entering={FadeInUp.delay(250 + index * 20).springify()}>
                            <BrandRow
                                brand={item}
                                isLiked={likedBrands.includes(item.id)}
                                onToggle={() => toggleBrand(item.id)}
                            />
                        </Animated.View>
                    )}
                    contentContainerStyle={styles.listContent}
                    showsVerticalScrollIndicator={false}
                />

                {/* Bottom CTA */}
                <Animated.View
                    entering={FadeInUp.delay(300).springify()}
                    style={styles.ctaContainer}
                >
                    <TouchableOpacity
                        style={[
                            styles.ctaButton,
                            !canContinue && styles.ctaButtonDisabled
                        ]}
                        onPress={handleContinue}
                        activeOpacity={canContinue ? 0.9 : 1}
                    >
                        <Text style={[
                            styles.ctaText,
                            !canContinue && styles.ctaTextDisabled
                        ]}>
                            {canContinue
                                ? 'Continue'
                                : `Like at least ${3 - likedBrands.length} more brand${3 - likedBrands.length > 1 ? 's' : ''}`
                            }
                        </Text>
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    progressContainer: {
        flexDirection: 'row',
        gap: 8,
        flex: 1,
        marginRight: 20,
    },
    progressStep: {
        flex: 1,
        height: 3,
        borderRadius: 1.5,
    },
    progressStepActive: {
        backgroundColor: '#000',
    },
    progressStepInactive: {
        backgroundColor: '#E5E7EB',
    },
    skipButton: {
        paddingHorizontal: 8,
    },
    skipText: {
        fontSize: 14,
        fontWeight: '500',
        color: '#000',
    },
    iconContainer: {
        paddingHorizontal: 20,
        marginTop: 16,
    },
    tagIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: '#F3F4F6',
        alignItems: 'center',
        justifyContent: 'center',
    },
    titleContainer: {
        paddingHorizontal: 20,
        marginTop: 20,
        marginBottom: 24,
    },
    title: {
        fontSize: 32,
        fontWeight: '600',
        color: '#000',
        lineHeight: 40,
        marginBottom: 12,
    },
    titleItalic: {
        fontStyle: 'italic',
    },
    subtitle: {
        fontSize: 15,
        color: '#6B7280',
        lineHeight: 22,
    },
    searchContainer: {
        marginHorizontal: 20,
        marginBottom: 16,
        position: 'relative',
    },
    searchInput: {
        backgroundColor: '#F3F4F6',
        borderRadius: 12,
        paddingHorizontal: 16,
        paddingVertical: 14,
        paddingRight: 44,
        fontSize: 15,
        color: '#000',
    },
    searchIcon: {
        position: 'absolute',
        right: 16,
        top: 14,
    },
    listContent: {
        paddingHorizontal: 20,
        paddingBottom: 100,
    },
    brandRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#F3F4F6',
    },
    brandName: {
        fontSize: 16,
        fontWeight: '400',
        color: '#000',
    },
    ctaContainer: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingHorizontal: 20,
        paddingVertical: 20,
        backgroundColor: '#fff',
    },
    ctaButton: {
        backgroundColor: '#000',
        paddingVertical: 16,
        borderRadius: 12,
        alignItems: 'center',
    },
    ctaButtonDisabled: {
        backgroundColor: '#1F2937',
    },
    ctaText: {
        fontSize: 16,
        fontWeight: '600',
        color: '#fff',
    },
    ctaTextDisabled: {
        color: 'rgba(255,255,255,0.7)',
    },
});

export default BrandSelectionScreen;
