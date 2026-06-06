import React, { useState } from 'react';
import { ActivityIndicator, TouchableOpacity, View, StyleSheet } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { useAdminGuard } from '../hooks/useAdminGuard';
import { useTranslation } from 'react-i18next';
import { AdminAddTab } from './admin/AdminAddTab';
import { AdminManageTab } from './admin/AdminManageTab';
import { AdminOccasionsTab } from './admin/AdminOccasionsTab';
import { AdminInspoTab } from './admin/AdminInspoTab';
import { AdminGuideTab } from './admin/AdminGuideTab';

type AdminTab = 'add' | 'manage' | 'occasions' | 'inspo' | 'guide';

const AdminPanelScreen = () => {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const { isAdmin, loading } = useAdminGuard();
    const [activeTab, setActiveTab] = useState<AdminTab>('add');

    if (loading) {
        return <SafeAreaView style={s.center}><ActivityIndicator size="large" color="#007AFF" /></SafeAreaView>;
    }
    if (!isAdmin) {
        return (
            <SafeAreaView style={s.center}>
                <Ionicons name="lock-closed" size={48} color="#FF3B30" />
                <ScaledText style={s.noAccess}>{t('admin.accessDenied')}</ScaledText>
                <ScaledText style={s.noAccessSub}>{t('admin.adminPrivilegesRequired')}</ScaledText>
            </SafeAreaView>
        );
    }

    const tabs: { key: AdminTab; icon: string; label: string }[] = [
        { key: 'add', icon: 'add-circle', label: t('admin.tabs.add') },
        { key: 'manage', icon: 'list', label: t('admin.tabs.manage') },
        { key: 'occasions', icon: 'shirt', label: t('admin.tabs.occasions', 'Occasions') },
        { key: 'inspo', icon: 'images', label: t('admin.tabs.inspo') },
        { key: 'guide', icon: 'book', label: t('admin.tabs.guide') },
    ];

    return (
        <SafeAreaView style={s.container} edges={['top', 'bottom']}>
            <View style={s.header}>
                <View style={s.headerRow}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={s.backBtn}>
                        <Ionicons name="chevron-back" size={28} color="#007AFF" />
                    </TouchableOpacity>
                    <View>
                        <ScaledText style={s.headerTitle}>{t('admin.title')}</ScaledText>
                        <ScaledText style={s.headerSub}>{t('admin.subtitle')}</ScaledText>
                    </View>
                </View>
            </View>
            <View style={s.tabBar}>
                {tabs.map((tab) => (
                    <TouchableOpacity key={tab.key} style={[s.tabBtn, activeTab === tab.key && s.tabBtnActive]} onPress={() => setActiveTab(tab.key)}>
                        <Ionicons name={tab.icon as any} size={20} color={activeTab === tab.key ? '#007AFF' : '#8E8E93'} />
                        <ScaledText style={[s.tabLabel, activeTab === tab.key && s.tabLabelActive]}>{tab.label}</ScaledText>
                    </TouchableOpacity>
                ))}
            </View>
            {activeTab === 'add' && <AdminAddTab />}
            {activeTab === 'manage' && <AdminManageTab />}
            {activeTab === 'occasions' && <AdminOccasionsTab />}
            {activeTab === 'inspo' && <AdminInspoTab />}
            {activeTab === 'guide' && <AdminGuideTab />}
        </SafeAreaView>
    );
};

const s = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F2F2F7' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F2F2F7' },
    noAccess: { fontSize: 18, fontWeight: '600', color: '#FF3B30', marginTop: 12 },
    noAccessSub: { fontSize: 14, color: '#8E8E93', marginTop: 4 },
    header: { paddingHorizontal: 16, paddingTop: 4, paddingBottom: 12 },
    headerRow: { flexDirection: 'row', alignItems: 'center' },
    backBtn: { marginRight: 8, padding: 4 },
    headerTitle: { fontSize: 28, fontWeight: '700', color: '#1C1C1E' },
    headerSub: { fontSize: 15, color: '#8E8E93', marginTop: 2 },
    tabBar: { flexDirection: 'row', marginHorizontal: 16, marginBottom: 8, backgroundColor: '#E5E5EA', borderRadius: 12, padding: 3 },
    tabBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 8, borderRadius: 10 },
    tabBtnActive: { backgroundColor: '#FFF', shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 4, elevation: 2 },
    tabLabel: { fontSize: 13, fontWeight: '500', color: '#8E8E93', marginLeft: 4 },
    tabLabelActive: { color: '#007AFF' },
});

export default AdminPanelScreen;
