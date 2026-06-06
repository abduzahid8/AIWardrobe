import React, { useState } from 'react';
import { View, StyleSheet, Dimensions, TouchableOpacity, ScrollView, StatusBar } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn, FadeInDown, FadeInUp, SlideInRight,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import {
    STYLE_PERSONALITIES, COLOR_OPTIONS, OCCASIONS, FIT_OPTIONS, STYLE_GOALS,
} from '../features/style-quiz/data';
import { useTranslation } from 'react-i18next';

const AnimatedScaledText = Animated.createAnimatedComponent(ScaledText);

const { width: W } = Dimensions.get('window');

// ── Design tokens ────────────────────────────────────────────────────────────
const T = {
    bg:           '#EEF4FF',
    surface:      'rgba(255, 255, 255, 0.85)',
    card:         '#FFFFFF',
    border:       'rgba(24, 58, 103, 0.08)',
    borderActive: '#0A1931',
    primary:      '#0A1931',   // Innovation Blue
    accent:       '#0A1931',
    accentLight:  'rgba(10, 25, 49, 0.08)',
    accent2:      '#254F86',
    text:         '#0A1931',
    sub:          '#4D4D4D',
    muted:        '#808080',
};

// ── Dot step indicator ───────────────────────────────────────────────────────
const StepDots = ({ total, current }: { total: number; current: number }) => (
    <View style={dot.row}>
        {Array.from({ length: total }).map((_, i) => (
            <Animated.View
                key={i}
                entering={FadeIn}
                style={[dot.d, i === current && dot.active, i < current && dot.done]}
            />
        ))}
    </View>
);
const dot = StyleSheet.create({
    row:    { flexDirection: 'row', gap: 6, alignItems: 'center' },
    d:      { width: 6, height: 6, borderRadius: 3, backgroundColor: 'rgba(10, 25, 49, 0.2)' },
    active: { width: 20, backgroundColor: T.primary },
    done:   { backgroundColor: T.accent2 },
});

// ── Shared button row ────────────────────────────────────────────────────────
const NavRow = ({
    onBack, onNext, disabled, label, isLast,
}: { onBack: () => void; onNext: () => void; disabled?: boolean; label?: string; isLast?: boolean }) => (
    <View style={nav.row}>
        <TouchableOpacity onPress={onBack} style={nav.back} activeOpacity={0.7}>
            <Ionicons name="chevron-back" size={22} color={T.primary} />
        </TouchableOpacity>
        <TouchableOpacity
            onPress={onNext}
            disabled={disabled}
            activeOpacity={0.85}
            style={[nav.next, disabled && nav.disabled]}
        >
            <LinearGradient
                colors={disabled ? ['#E2E8F0', '#CBD5E1'] : ['#0A1931', '#1a3a5c']}
                start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
                style={nav.gradient}
            >
                <ScaledText style={[nav.label, disabled && { color: '#94A3B8' }]}>{label ?? 'Continue'}</ScaledText>
                <Ionicons name={isLast ? 'checkmark' : 'arrow-forward'} size={18} color={disabled ? '#94A3B8' : '#FFF'} />
            </LinearGradient>
        </TouchableOpacity>
    </View>
);
const nav = StyleSheet.create({
    row:      { flexDirection: 'row', gap: 12, paddingTop: 16 },
    back:     { width: 52, height: 52, borderRadius: 26, borderWidth: 1, borderColor: T.border, alignItems: 'center', justifyContent: 'center', backgroundColor: '#FFF' },
    next:     { flex: 1, borderRadius: 26, overflow: 'hidden' },
    gradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 15, gap: 8 },
    label:    { fontSize: 16, fontWeight: '600', color: '#FFF', letterSpacing: 0.2 },
    disabled: { opacity: 0.5 },
});

// ── WELCOME ──────────────────────────────────────────────────────────────────
const WelcomeStep = ({ onNext, t }: { onNext: () => void; t: any }) => (
    <Animated.View style={s.step} entering={FadeIn.duration(600)}>
        <View style={{ flex: 1, justifyContent: 'center' }}>
            <Animated.View entering={FadeInDown.delay(100).duration(700)}>
                <LinearGradient
                    colors={['#0A1931','#254F86','#EEF4FF']}
                    start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
                    style={wlc.orb}
                />
            </Animated.View>

            <AnimatedScaledText entering={FadeInDown.delay(200).duration(700)} style={wlc.headline}>
                {t('styleQuiz.welcome.title')}
            </AnimatedScaledText>
            <AnimatedScaledText entering={FadeInDown.delay(350).duration(700)} style={wlc.sub}>
                {t('styleQuiz.welcome.subtitle')}
            </AnimatedScaledText>

            <Animated.View entering={FadeInDown.delay(500).duration(700)} style={wlc.pills}>
                {[
                    'styleQuiz.welcome.aiPlans',
                    'styleQuiz.welcome.colorProfiling',
                    'styleQuiz.welcome.gapAnalysis',
                ].map(key => (
                    <View key={key} style={wlc.pill}>
                        <Ionicons name="sparkles" size={13} color={T.accent} />
                        <ScaledText style={wlc.pillTxt}>{t(key)}</ScaledText>
                    </View>
                ))}
            </Animated.View>
        </View>

        <Animated.View entering={FadeInUp.delay(600).duration(700)}>
            <TouchableOpacity onPress={onNext} activeOpacity={0.85}>
                <LinearGradient
                    colors={['#0A1931','#1a3a5c']}
                    start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
                    style={wlc.cta}
                >
                    <ScaledText style={wlc.ctaTxt}>{t('styleQuiz.continue')}</ScaledText>
                    <Ionicons name="arrow-forward" size={20} color="#FFF" />
                </LinearGradient>
            </TouchableOpacity>
        </Animated.View>
    </Animated.View>
);
const wlc = StyleSheet.create({
    orb:      { width: 120, height: 120, borderRadius: 60, marginBottom: 40, opacity: 0.85 },
    headline: { fontSize: 44, fontWeight: '800', color: T.text, letterSpacing: -1.5, lineHeight: 52, marginBottom: 16 },
    sub:      { fontSize: 17, color: T.sub, lineHeight: 26, marginBottom: 36 },
    pills:    { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
    pill:     { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(10,25,49,0.05)', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: 'rgba(10,25,49,0.12)' },
    pillTxt:  { fontSize: 13, color: T.primary, fontWeight: '600' },
    cta:      { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 18, borderRadius: 28, gap: 10 },
    ctaTxt:   { fontSize: 18, fontWeight: '700', color: '#FFF' },
});

// ── PERSONALITY ──────────────────────────────────────────────────────────────
const PersonalityStep = ({ selected, onSelect, onNext, onBack, t }: any) => (
    <Animated.View style={s.step} entering={SlideInRight.duration(350).springify()}>
        <ScaledText style={s.title}>{t('styleQuiz.personality.title')}</ScaledText>
        <ScaledText style={s.sub}>{t('styleQuiz.personality.subtitle')}</ScaledText>
        <ScrollView showsVerticalScrollIndicator={false} style={{ flex: 1 }}>
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 12, paddingBottom: 16 }}>
                {STYLE_PERSONALITIES.map(item => {
                    const on = selected === item.id;
                    return (
                        <TouchableOpacity
                            key={item.id}
                            onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onSelect(item.id); }}
                            activeOpacity={0.8}
                            style={[pc.card, { width: (W - 60) / 2 }, on && pc.sel]}
                        >
                            <ScaledText style={pc.emoji}>{item.emoji}</ScaledText>
                            <ScaledText style={[pc.name, on && { color: T.accent }]}>{t(item.tKey)}</ScaledText>
                            <ScaledText style={pc.desc}>{t(item.tDescKey)}</ScaledText>
                            {on && <View style={pc.check}><Ionicons name="checkmark" size={12} color="#FFF" /></View>}
                        </TouchableOpacity>
                    );
                })}
            </View>
        </ScrollView>
        <NavRow onBack={onBack} onNext={onNext} disabled={!selected} />
    </Animated.View>
);
const pc = StyleSheet.create({
    card:  { backgroundColor: 'rgba(255, 255, 255, 0.85)', borderRadius: 20, padding: 18, borderWidth: 1, borderColor: T.border, shadowColor: '#173A65', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.04, shadowRadius: 12, elevation: 2 },
    sel:   { borderColor: '#0A1931', borderWidth: 1.5, backgroundColor: 'rgba(10, 25, 49, 0.06)' },
    emoji: { fontSize: 34, marginBottom: 10 },
    name:  { fontSize: 15, fontWeight: '700', color: T.text, marginBottom: 4 },
    desc:  { fontSize: 12, color: T.sub, lineHeight: 17 },
    check: { position: 'absolute', top: 10, right: 10, width: 22, height: 22, borderRadius: 11, backgroundColor: '#0A1931', alignItems: 'center', justifyContent: 'center' },
});

// ── COLORS ────────────────────────────────────────────────────────────────────
const ColorsStep = ({ favoriteColors, onToggleColor, onNext, onBack, t }: any) => (
    <Animated.View style={s.step} entering={SlideInRight.duration(350).springify()}>
        <ScaledText style={s.title}>{t('styleQuiz.colors.title')}</ScaledText>
        <ScaledText style={s.sub}>{t('styleQuiz.colors.subtitle', { count: favoriteColors.length })}</ScaledText>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 16, justifyContent: 'center', flex: 1, alignContent: 'center' }}>
            {COLOR_OPTIONS.map(c => {
                const on = favoriteColors.includes(c.id);
                return (
                    <TouchableOpacity
                        key={c.id}
                        onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onToggleColor(c.id); }}
                        activeOpacity={0.8}
                        style={[col.swatch, { backgroundColor: c.color }, on && col.ring]}
                    >
                        {on && <Ionicons name="checkmark" size={20} color={c.id === 'white' || c.id === 'beige' ? '#000' : '#FFF'} />}
                    </TouchableOpacity>
                );
            })}
        </View>
        <NavRow onBack={onBack} onNext={onNext} disabled={favoriteColors.length < 1} />
    </Animated.View>
);
const col = StyleSheet.create({
    swatch: { width: 62, height: 62, borderRadius: 31, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: 'transparent' },
    ring:   { borderColor: '#0A1931', borderWidth: 3 },
});

// ── OCCASIONS ─────────────────────────────────────────────────────────────────
const OccasionsStep = ({ selectedOccasions, onToggleOccasion, onNext, onBack, t }: any) => (
    <Animated.View style={s.step} entering={SlideInRight.duration(350).springify()}>
        <ScaledText style={s.title}>{t('styleQuiz.occasions.title')}</ScaledText>
        <ScaledText style={s.sub}>{t('styleQuiz.occasions.subtitle')}</ScaledText>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 12, flex: 1, alignContent: 'center' }}>
            {OCCASIONS.map(o => {
                const on = selectedOccasions.includes(o.id);
                return (
                    <TouchableOpacity
                        key={o.id}
                        onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onToggleOccasion(o.id); }}
                        activeOpacity={0.8}
                        style={[occ.card, { width: (W - 60) / 2 }, on && occ.sel]}
                    >
                        <View style={[occ.icon, on && occ.iconSel]}>
                            <Ionicons name={o.icon as any} size={26} color={on ? '#FFF' : T.primary} />
                        </View>
                        <ScaledText style={[occ.name, on && { color: T.text, fontWeight: '700' }]}>{t(o.tKey)}</ScaledText>
                    </TouchableOpacity>
                );
            })}
        </View>
        <NavRow onBack={onBack} onNext={onNext} disabled={selectedOccasions.length < 1} />
    </Animated.View>
);
const occ = StyleSheet.create({
    card:    { backgroundColor: 'rgba(255, 255, 255, 0.85)', borderRadius: 20, padding: 18, alignItems: 'center', borderWidth: 1, borderColor: T.border, shadowColor: '#173A65', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.04, shadowRadius: 12, elevation: 2 },
    sel:     { borderColor: '#0A1931', borderWidth: 1.5, backgroundColor: 'rgba(10, 25, 49, 0.06)' },
    icon:    { width: 54, height: 54, borderRadius: 27, backgroundColor: 'rgba(10, 25, 49, 0.05)', alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
    iconSel: { backgroundColor: '#0A1931' },
    name:    { fontSize: 13, color: T.primary, textAlign: 'center' },
});

// ── FIT ───────────────────────────────────────────────────────────────────────
const FitStep = ({ selected, onSelect, onNext, onBack, t }: any) => (
    <Animated.View style={s.step} entering={SlideInRight.duration(350).springify()}>
        <ScaledText style={s.title}>{t('styleQuiz.fit.title')}</ScaledText>
        <ScaledText style={s.sub}>{t('styleQuiz.fit.subtitle')}</ScaledText>
        <View style={{ gap: 14, flex: 1, justifyContent: 'center' }}>
            {FIT_OPTIONS.map(f => {
                const on = selected === f.id;
                return (
                    <TouchableOpacity
                        key={f.id}
                        onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onSelect(f.id); }}
                        activeOpacity={0.8}
                        style={[fit.card, on && fit.sel]}
                    >
                        <ScaledText style={fit.emoji}>{f.icon}</ScaledText>
                        <View style={{ flex: 1 }}>
                            <ScaledText style={[fit.name, on && { color: T.accent }]}>{t(f.tKey)}</ScaledText>
                            <ScaledText style={fit.desc}>{t(f.tDescKey)}</ScaledText>
                        </View>
                        {on && <Ionicons name="checkmark-circle" size={24} color={T.accent} />}
                    </TouchableOpacity>
                );
            })}
        </View>
        <NavRow onBack={onBack} onNext={onNext} />
    </Animated.View>
);
const fit = StyleSheet.create({
    card:  { flexDirection: 'row', alignItems: 'center', gap: 16, backgroundColor: 'rgba(255, 255, 255, 0.85)', borderRadius: 20, padding: 20, borderWidth: 1, borderColor: T.border, shadowColor: '#173A65', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.04, shadowRadius: 12, elevation: 2 },
    sel:   { borderColor: '#0A1931', borderWidth: 1.5, backgroundColor: 'rgba(10, 25, 49, 0.06)' },
    emoji: { fontSize: 32 },
    name:  { fontSize: 17, fontWeight: '700', color: T.text, marginBottom: 3 },
    desc:  { fontSize: 13, color: T.sub },
});

// ── GOALS ─────────────────────────────────────────────────────────────────────
const GoalsStep = ({ selectedGoals, onToggleGoal, onNext, onBack, t }: any) => (
    <Animated.View style={s.step} entering={SlideInRight.duration(350).springify()}>
        <ScaledText style={s.title}>{t('styleQuiz.goals.title')}</ScaledText>
        <ScaledText style={s.sub}>{t('styleQuiz.goals.subtitle')}</ScaledText>
        <ScrollView showsVerticalScrollIndicator={false} style={{ flex: 1 }}>
            <View style={{ gap: 10, paddingBottom: 16 }}>
                {STYLE_GOALS.map(g => {
                    const on = selectedGoals.includes(g.id);
                    return (
                        <TouchableOpacity
                            key={g.id}
                            onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onToggleGoal(g.id); }}
                            activeOpacity={0.8}
                            style={[gls.card, on && gls.sel]}
                        >
                            <View style={[gls.icon, on && gls.iconSel]}>
                                <Ionicons name={g.icon as any} size={20} color={on ? '#FFF' : T.primary} />
                            </View>
                            <ScaledText style={[gls.name, on && { color: T.text, fontWeight: '700' }]}>{t(g.tKey)}</ScaledText>
                            {on && <Ionicons name="checkmark-circle" size={20} color={T.accent} />}
                        </TouchableOpacity>
                    );
                })}
            </View>
        </ScrollView>
        <NavRow onBack={onBack} onNext={onNext} />
    </Animated.View>
);
const gls = StyleSheet.create({
    card:    { flexDirection: 'row', alignItems: 'center', gap: 14, backgroundColor: 'rgba(255, 255, 255, 0.85)', borderRadius: 18, padding: 16, borderWidth: 1, borderColor: T.border, shadowColor: '#173A65', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.04, shadowRadius: 12, elevation: 2 },
    sel:     { borderColor: '#0A1931', borderWidth: 1.5, backgroundColor: 'rgba(10, 25, 49, 0.06)' },
    icon:    { width: 42, height: 42, borderRadius: 21, backgroundColor: 'rgba(10, 25, 49, 0.05)', alignItems: 'center', justifyContent: 'center' },
    iconSel: { backgroundColor: '#0A1931' },
    name:    { flex: 1, fontSize: 15, color: T.primary },
});

// ── COMPLETE ───────────────────────────────────────────────────────────────────
const CompleteStep = ({ selectedGoals, selectedOccasions, onComplete, onBack, t }: any) => (
    <Animated.View style={s.step} entering={FadeIn.duration(600)}>
        <View style={{ flex: 1, justifyContent: 'center' }}>
            <Animated.View entering={FadeInDown.delay(100)} style={cmp.badge}>
                <LinearGradient colors={['#0A1931','#254F86']} style={cmp.badgeInner}>
                    <Ionicons name="checkmark" size={40} color="#FFF" />
                </LinearGradient>
            </Animated.View>

            <AnimatedScaledText entering={FadeInDown.delay(250)} style={cmp.title}>
                {t('styleQuiz.complete.title')}
            </AnimatedScaledText>
            <AnimatedScaledText entering={FadeInDown.delay(400)} style={cmp.sub}>
                {t('styleQuiz.complete.subtitle')}
            </AnimatedScaledText>

            <Animated.View entering={FadeInDown.delay(550)} style={{ gap: 12, marginTop: 32 }}>
                {[
                    t('styleQuiz.complete.occasionPaths'),
                    t('styleQuiz.complete.goalsPrioritized'),
                    t('styleQuiz.complete.aiCalibrated'),
                ].map(line => (
                    <View key={line} style={cmp.row}>
                        <View style={cmp.dot} />
                        <ScaledText style={cmp.rowTxt}>{line}</ScaledText>
                    </View>
                ))}
            </Animated.View>
        </View>

        <NavRow onBack={onBack} onNext={onComplete} label={t('styleQuiz.complete.unlockPlan')} isLast />
    </Animated.View>
);
const cmp = StyleSheet.create({
    badge:      { alignSelf: 'flex-start', marginBottom: 32 },
    badgeInner: { width: 80, height: 80, borderRadius: 40, alignItems: 'center', justifyContent: 'center' },
    title:      { fontSize: 40, fontWeight: '800', color: T.text, letterSpacing: -1.2, lineHeight: 48, marginBottom: 14 },
    sub:        { fontSize: 16, color: T.sub, lineHeight: 25 },
    row:        { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: 'rgba(255, 255, 255, 0.85)', borderRadius: 14, paddingHorizontal: 18, paddingVertical: 14, borderWidth: 1, borderColor: T.border, shadowColor: '#173A65', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.03, shadowRadius: 8, elevation: 1 },
    dot:        { width: 8, height: 8, borderRadius: 4, backgroundColor: T.accent2 },
    rowTxt:     { fontSize: 15, color: T.text, fontWeight: '500' },
});

// ── MAIN SCREEN ───────────────────────────────────────────────────────────────
const STEPS = ['Welcome','Personality','Colors','Occasions','Fit','Goals','Done'];

const StyleQuizScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const { setPreferences, setOnboardingStep, completeOnboarding } = useStylePreferenceStore();

    const [step, setStep] = useState(0);
    const [stylePersonality, setStylePersonality] = useState<string | undefined>();
    const [favoriteColors, setFavoriteColors] = useState<string[]>([]);
    const [occasions, setOccasions] = useState<string[]>([]);
    const [fitPreference, setFitPreference] = useState<string>('balanced');
    const [goals, setGoals] = useState<string[]>([]);

    const next = () => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); setStep(p => p + 1); };
    const back = () => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setStep(p => Math.max(0, p - 1)); };

    const toggleColor   = (id: string) => setFavoriteColors(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);
    const toggleOccasion = (id: string) => setOccasions(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);
    const toggleGoal    = (id: string) => setGoals(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);

    const handleComplete = () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        setPreferences({
            stylePersonality: stylePersonality as any,
            favoriteColors,
            primaryOccasions: occasions,
            fitPreference: fitPreference as any,
            styleGoals: goals,
            prefersSustainable: goals.includes('sustainability'),
        });
        completeOnboarding();
    };

    const renderStep = () => {
        switch (step) {
            case 0: return <WelcomeStep onNext={next} t={t} />;
            case 1: return <PersonalityStep selected={stylePersonality} onSelect={setStylePersonality} onNext={next} onBack={back} t={t} />;
            case 2: return <ColorsStep favoriteColors={favoriteColors} onToggleColor={toggleColor} onNext={next} onBack={back} t={t} />;
            case 3: return <OccasionsStep selectedOccasions={occasions} onToggleOccasion={toggleOccasion} onNext={next} onBack={back} t={t} />;
            case 4: return <FitStep selected={fitPreference} onSelect={setFitPreference} onNext={next} onBack={back} t={t} />;
            case 5: return <GoalsStep selectedGoals={goals} onToggleGoal={toggleGoal} onNext={next} onBack={back} t={t} />;
            case 6: return <CompleteStep selectedGoals={goals} selectedOccasions={occasions} onComplete={handleComplete} onBack={back} t={t} />;
            default: return null;
        }
    };

    return (
        <View style={s.root}>
            <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />
            {/* Background gradient */}
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
            />
            {/* Ambient light orbs */}
            <View style={s.backgroundOrbTop} pointerEvents="none" />
            <View style={s.backgroundOrbBottom} pointerEvents="none" />

            <SafeAreaView style={s.safe}>
                {/* Header */}
                <View style={s.header}>
                    {step > 0
                        ? <StepDots total={STEPS.length - 1} current={step - 1} />
                        : <View style={{ height: 6 }} />
                    }
                </View>

                {renderStep()}
            </SafeAreaView>
        </View>
    );
};

// ── Root styles ───────────────────────────────────────────────────────────────
const s = StyleSheet.create({
    root:   { flex: 1, backgroundColor: T.bg },
    safe:   { flex: 1 },
    backgroundOrbTop: { position: 'absolute', top: -100, right: -80, width: 280, height: 280, borderRadius: 140, backgroundColor: 'rgba(188, 210, 245, 0.42)' },
    backgroundOrbBottom: { position: 'absolute', left: -120, bottom: 140, width: 300, height: 300, borderRadius: 150, backgroundColor: 'rgba(216, 229, 252, 0.34)' },
    header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 24, paddingTop: 12, paddingBottom: 8, minHeight: 40 },
    step:   { flex: 1, paddingHorizontal: 24, paddingBottom: 24 },
    title:  { fontSize: 36, fontWeight: '800', color: T.text, letterSpacing: -1.2, lineHeight: 44, marginTop: 24, marginBottom: 10 },
    sub:    { fontSize: 15, color: T.sub, marginBottom: 28, lineHeight: 23 },
});

export default StyleQuizScreen;
