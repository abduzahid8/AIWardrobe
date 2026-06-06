/**
 * AdminUsersTab — Manage admin roles and permissions
 * Add new Gmail accounts as admins
 */

import React, { useState, useCallback, useEffect } from 'react';
import { ActivityIndicator, Alert, FlatList, Modal, ScrollView, StyleSheet, TextInput, TouchableOpacity, View,  } from 'react-native'
import { ScaledText } from '../../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTranslation } from 'react-i18next';
import { createLogger } from '../../src/utils/logger';
import {
    getAllUsers,
    assignAdminRole,
    revokeAdminRole,
    AdminUser,
    AdminRole,
} from '../../src/services/adminService';

const logger = createLogger('AdminUsersTab');

export const AdminUsersTab = () => {
    const { t } = useTranslation();
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [searchText, setSearchText] = useState('');
    const [showAddModal, setShowAddModal] = useState(false);
    const [newAdminEmail, setNewAdminEmail] = useState('');
    const [selectedRole, setSelectedRole] = useState<AdminRole>('admin');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const ADMIN_ROLES: { value: AdminRole; label: string }[] = [
        { value: 'admin', label: 'Admin' },
        { value: 'moderator', label: 'Moderator' },
        { value: 'super_admin', label: 'Super Admin' },
    ];

    const fetchUsers = useCallback(async () => {
        try {
            setLoading(true);
            setError('');
            const data = await getAllUsers();
            setUsers(data);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load users';
            logger.error('Failed to fetch users:', err);
            setError(message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchUsers();
    }, [fetchUsers]);

    const handleRefresh = async () => {
        setRefreshing(true);
        await fetchUsers();
        setRefreshing(false);
    };

    const handleAddAdmin = async () => {
        if (!newAdminEmail.trim()) {
            setError('Please enter an email address');
            return;
        }

        if (!newAdminEmail.includes('@')) {
            setError('Please enter a valid email address');
            return;
        }

        try {
            setIsSubmitting(true);
            setError('');
            await assignAdminRole(newAdminEmail, selectedRole);
            setSuccess(`✓ Admin role '${selectedRole}' assigned to ${newAdminEmail}`);
            setNewAdminEmail('');
            setSelectedRole('admin');
            setShowAddModal(false);
            await fetchUsers();
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to assign admin role';
            logger.error('Failed to assign admin role:', err);
            setError(message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleRevokeAdmin = (user: AdminUser) => {
        Alert.alert(
            'Revoke Admin Access',
            `Remove admin privileges from ${user.email}?`,
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Revoke',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            setLoading(true);
                            await revokeAdminRole(user.email);
                            setSuccess(`✓ Admin access revoked from ${user.email}`);
                            await fetchUsers();
                            setTimeout(() => setSuccess(''), 3000);
                        } catch (err) {
                            const message = err instanceof Error ? err.message : 'Failed to revoke admin role';
                            logger.error('Failed to revoke admin role:', err);
                            setError(message);
                        } finally {
                            setLoading(false);
                        }
                    },
                },
            ]
        );
    };

    const filteredUsers = users.filter(
        (user) =>
            user.email.toLowerCase().includes(searchText.toLowerCase()) ||
            user.username.toLowerCase().includes(searchText.toLowerCase())
    );

    const adminUsers = filteredUsers.filter((u) => u.is_admin);
    const regularUsers = filteredUsers.filter((u) => !u.is_admin);

    const renderUserCard = (user: AdminUser, isAdmin: boolean) => (
        <View key={user.id} style={[s.userCard, isAdmin && s.userCardAdmin]}>
            <View style={s.userInfo}>
                <ScaledText style={s.userEmail}>{user.email}</ScaledText>
                <ScaledText style={s.userName}>{user.username}</ScaledText>
                {isAdmin && (
                    <View style={s.roleBadge}>
                        <ScaledText style={s.roleBadgeText}>{user.admin_role}</ScaledText>
                    </View>
                )}
                <ScaledText style={s.userDate}>
                    {isAdmin && user.admin_assigned_at
                        ? `Admin since ${new Date(user.admin_assigned_at).toLocaleDateString()}`
                        : `Joined ${new Date(user.created_at).toLocaleDateString()}`}
                </ScaledText>
            </View>
            {isAdmin && (
                <TouchableOpacity
                    style={s.revokeBtn}
                    onPress={() => handleRevokeAdmin(user)}
                >
                    <Ionicons name="close-circle" size={24} color="#FF3B30" />
                </TouchableOpacity>
            )}
        </View>
    );

    return (
        <View style={s.container}>
            {/* Search Bar */}
            <View style={s.searchContainer}>
                <Ionicons name="search" size={18} color="#8E8E93" style={s.searchIcon} />
                <TextInput
                    style={s.searchInput}
                    placeholder="Search by email or username..."
                    value={searchText}
                    onChangeText={setSearchText}
                    placeholderTextColor="#8E8E93"
                />
                {searchText.length > 0 && (
                    <TouchableOpacity onPress={() => setSearchText('')}>
                        <Ionicons name="close-circle" size={18} color="#8E8E93" />
                    </TouchableOpacity>
                )}
            </View>

            {/* Add Admin Button */}
            <TouchableOpacity
                style={s.addButton}
                onPress={() => {
                    setShowAddModal(true);
                    setError('');
                }}
            >
                <Ionicons name="add-circle" size={20} color="#FFF" />
                <ScaledText style={s.addButtonText}>Add New Admin</ScaledText>
            </TouchableOpacity>

            {/* Error Message */}
            {error && (
                <View style={s.errorBanner}>
                    <Ionicons name="alert-circle" size={18} color="#FF3B30" />
                    <ScaledText style={s.errorText}>{error}</ScaledText>
                </View>
            )}

            {/* Success Message */}
            {success && (
                <View style={s.successBanner}>
                    <Ionicons name="checkmark-circle" size={18} color="#34C759" />
                    <ScaledText style={s.successText}>{success}</ScaledText>
                </View>
            )}

            {/* Content */}
            {loading ? (
                <View style={s.center}>
                    <ActivityIndicator size="large" color="#007AFF" />
                </View>
            ) : users.length === 0 ? (
                <View style={s.center}>
                    <Ionicons name="people-outline" size={48} color="#8E8E93" />
                    <ScaledText style={s.emptyText}>No users found</ScaledText>
                </View>
            ) : (
                <FlatList
                    data={[
                        ...(adminUsers.length > 0
                            ? [
                                { type: 'header' as const, title: `Admins (${adminUsers.length})` },
                                ...adminUsers.map((u) => ({ type: 'user' as const, user: u, isAdmin: true })),
                            ]
                            : []),
                        ...(regularUsers.length > 0
                            ? [
                                { type: 'header' as const, title: `Regular Users (${regularUsers.length})` },
                                ...regularUsers.map((u) => ({ type: 'user' as const, user: u, isAdmin: false })),
                            ]
                            : []),
                    ]}
                    keyExtractor={(item, idx) =>
                        item.type === 'header' ? `header-${idx}` : item.user?.id || `user-${idx}`
                    }
                    renderItem={({ item }) => {
                        if (item.type === 'header') {
                            return <ScaledText style={s.sectionHeader}>{item.title}</ScaledText>;
                        }
                        return renderUserCard(item.user, item.isAdmin);
                    }}
                    contentContainerStyle={{ paddingBottom: 40 }}
                    refreshing={refreshing}
                    onRefresh={handleRefresh}
                />
            )}

            {/* Add Admin Modal */}
            <Modal visible={showAddModal} animationType="slide" presentationStyle="pageSheet">
                <SafeAreaView style={s.modalContainer}>
                    <View style={s.modalHeader}>
                        <TouchableOpacity onPress={() => setShowAddModal(false)}>
                            <ScaledText style={s.modalCancel}>Cancel</ScaledText>
                        </TouchableOpacity>
                        <ScaledText style={s.modalTitle}>Add New Admin</ScaledText>
                        <TouchableOpacity
                            onPress={handleAddAdmin}
                            disabled={isSubmitting || !newAdminEmail.trim()}
                        >
                            <ScaledText
                                style={[
                                    s.modalSave,
                                    (isSubmitting || !newAdminEmail.trim()) && s.modalSaveDisabled,
                                ]}
                            >
                                {isSubmitting ? 'Adding...' : 'Add'}
                            </ScaledText>
                        </TouchableOpacity>
                    </View>

                    <ScrollView style={s.modalBody} keyboardShouldPersistTaps="handled">
                        <View style={s.formSection}>
                            <ScaledText style={s.label}>Email Address</ScaledText>
                            <TextInput
                                style={s.input}
                                placeholder="user@gmail.com"
                                value={newAdminEmail}
                                onChangeText={setNewAdminEmail}
                                keyboardType="email-address"
                                autoCapitalize="none"
                                editable={!isSubmitting}
                                placeholderTextColor="#8E8E93"
                            />
                            <ScaledText style={s.hint}>
                                Enter the Gmail address of the user you want to make an admin
                            </ScaledText>
                        </View>

                        <View style={s.formSection}>
                            <ScaledText style={s.label}>Admin Role</ScaledText>
                            <View style={s.roleOptions}>
                                {ADMIN_ROLES.map((role) => (
                                    <TouchableOpacity
                                        key={role.value}
                                        style={[
                                            s.roleOption,
                                            selectedRole === role.value && s.roleOptionActive,
                                        ]}
                                        onPress={() => setSelectedRole(role.value)}
                                        disabled={isSubmitting}
                                    >
                                        <View
                                            style={[
                                                s.roleRadio,
                                                selectedRole === role.value && s.roleRadioActive,
                                            ]}
                                        >
                                            {selectedRole === role.value && (
                                                <View style={s.roleRadioDot} />
                                            )}
                                        </View>
                                        <View style={{ flex: 1 }}>
                                            <ScaledText style={s.roleLabel}>{role.label}</ScaledText>
                                            <ScaledText style={s.roleDescription}>
                                                {role.value === 'super_admin'
                                                    ? 'Full system access, can assign admins'
                                                    : role.value === 'admin'
                                                    ? 'Can manage content and view logs'
                                                    : 'Limited access, view-only permissions'}
                                            </ScaledText>
                                        </View>
                                    </TouchableOpacity>
                                ))}
                            </View>
                        </View>

                        {error && (
                            <View style={s.errorBox}>
                                <Ionicons name="alert-circle" size={18} color="#FF3B30" />
                                <ScaledText style={s.errorBoxText}>{error}</ScaledText>
                            </View>
                        )}

                        <View style={{ height: 40 }} />
                    </ScrollView>
                </SafeAreaView>
            </Modal>
        </View>
    );
};

const s = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F2F2F7' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    emptyText: { fontSize: 16, color: '#8E8E93', marginTop: 12 },

    // Search
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginHorizontal: 16,
        marginVertical: 12,
        paddingHorizontal: 12,
        backgroundColor: '#FFF',
        borderRadius: 10,
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    searchIcon: { marginRight: 8 },
    searchInput: {
        flex: 1,
        paddingVertical: 10,
        fontSize: 15,
        color: '#1C1C1E',
    },

    // Add Button
    addButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        marginHorizontal: 16,
        marginBottom: 12,
        paddingVertical: 12,
        backgroundColor: '#007AFF',
        borderRadius: 10,
    },
    addButtonText: { fontSize: 16, fontWeight: '600', color: '#FFF', marginLeft: 8 },

    // Banners
    errorBanner: {
        flexDirection: 'row',
        alignItems: 'center',
        marginHorizontal: 16,
        marginBottom: 12,
        paddingHorizontal: 12,
        paddingVertical: 10,
        backgroundColor: '#FFE5E5',
        borderRadius: 10,
        borderLeftWidth: 4,
        borderLeftColor: '#FF3B30',
    },
    errorText: { fontSize: 13, color: '#FF3B30', marginLeft: 8, flex: 1 },
    successBanner: {
        flexDirection: 'row',
        alignItems: 'center',
        marginHorizontal: 16,
        marginBottom: 12,
        paddingHorizontal: 12,
        paddingVertical: 10,
        backgroundColor: '#E5F5E5',
        borderRadius: 10,
        borderLeftWidth: 4,
        borderLeftColor: '#34C759',
    },
    successText: { fontSize: 13, color: '#34C759', marginLeft: 8, flex: 1 },

    // User Card
    userCard: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginHorizontal: 16,
        marginVertical: 6,
        paddingHorizontal: 12,
        paddingVertical: 12,
        backgroundColor: '#FFF',
        borderRadius: 10,
        borderLeftWidth: 4,
        borderLeftColor: '#E5E5EA',
    },
    userCardAdmin: { borderLeftColor: '#F39C12' },
    userInfo: { flex: 1 },
    userEmail: { fontSize: 15, fontWeight: '600', color: '#1C1C1E' },
    userName: { fontSize: 13, color: '#8E8E93', marginTop: 2 },
    roleBadge: {
        alignSelf: 'flex-start',
        marginTop: 6,
        paddingHorizontal: 8,
        paddingVertical: 4,
        backgroundColor: '#F39C12',
        borderRadius: 6,
    },
    roleBadgeText: { fontSize: 11, fontWeight: '600', color: '#FFF', textTransform: 'capitalize' },
    userDate: { fontSize: 12, color: '#8E8E93', marginTop: 4 },
    revokeBtn: { padding: 8 },

    // Section Header
    sectionHeader: {
        fontSize: 14,
        fontWeight: '600',
        color: '#8E8E93',
        marginHorizontal: 16,
        marginTop: 16,
        marginBottom: 8,
        textTransform: 'uppercase',
    },

    // Modal
    modalContainer: { flex: 1, backgroundColor: '#F2F2F7' },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#E5E5EA',
        backgroundColor: '#FFF',
    },
    modalCancel: { fontSize: 16, color: '#007AFF' },
    modalTitle: { fontSize: 17, fontWeight: '600', color: '#1C1C1E' },
    modalSave: { fontSize: 16, fontWeight: '600', color: '#007AFF' },
    modalSaveDisabled: { color: '#8E8E93', opacity: 0.5 },
    modalBody: { flex: 1, paddingHorizontal: 16 },

    // Form
    formSection: { marginTop: 20 },
    label: { fontSize: 13, fontWeight: '600', color: '#636366', marginBottom: 8 },
    input: {
        backgroundColor: '#FFF',
        borderRadius: 10,
        paddingHorizontal: 14,
        paddingVertical: 12,
        fontSize: 15,
        color: '#1C1C1E',
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    hint: { fontSize: 12, color: '#8E8E93', marginTop: 6 },

    // Role Options
    roleOptions: { gap: 10 },
    roleOption: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        paddingHorizontal: 12,
        paddingVertical: 12,
        backgroundColor: '#FFF',
        borderRadius: 10,
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    roleOptionActive: { borderColor: '#007AFF', backgroundColor: '#F0F8FF' },
    roleRadio: {
        width: 20,
        height: 20,
        borderRadius: 10,
        borderWidth: 2,
        borderColor: '#8E8E93',
        marginRight: 12,
        marginTop: 2,
        justifyContent: 'center',
        alignItems: 'center',
    },
    roleRadioActive: { borderColor: '#007AFF' },
    roleRadioDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#007AFF' },
    roleLabel: { fontSize: 15, fontWeight: '600', color: '#1C1C1E' },
    roleDescription: { fontSize: 12, color: '#8E8E93', marginTop: 4 },

    // Error Box
    errorBox: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 16,
        paddingHorizontal: 12,
        paddingVertical: 10,
        backgroundColor: '#FFE5E5',
        borderRadius: 10,
        borderLeftWidth: 4,
        borderLeftColor: '#FF3B30',
    },
    errorBoxText: { fontSize: 13, color: '#FF3B30', marginLeft: 8, flex: 1 },
});
