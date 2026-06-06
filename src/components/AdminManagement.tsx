/**
 * Admin Management Component
 * 
 * UI for managing admin roles and permissions.
 * Allows super admins to assign/revoke admin roles and view audit logs.
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Alert,
    StyleSheet,
    FlatList,
    Modal,
} from 'react-native';
import {
    getAllUsers,
    getUserDetails,
    assignAdminRole,
    revokeAdminRole,
    getAdminLogs,
    getAdminStats,
    AdminUser,
    AdminLog,
    AdminStats,
    AdminRole,
} from '../services/adminService';
import { createLogger } from '../utils/logger';

const logger = createLogger('AdminManagement');
const ADMIN_ROLE_OPTIONS: Array<{ label: string; value: AdminRole }> = [
    { label: 'Admin', value: 'admin' },
    { label: 'Moderator', value: 'moderator' },
    { label: 'Super Admin', value: 'super_admin' },
];

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    header: {
        backgroundColor: '#2c3e50',
        padding: 16,
        paddingTop: 24,
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#fff',
    },
    section: {
        backgroundColor: '#fff',
        margin: 12,
        borderRadius: 8,
        padding: 16,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        marginBottom: 12,
        color: '#2c3e50',
    },
    inputGroup: {
        marginBottom: 12,
    },
    label: {
        fontSize: 14,
        fontWeight: '500',
        marginBottom: 6,
        color: '#34495e',
    },
    input: {
        borderWidth: 1,
        borderColor: '#bdc3c7',
        borderRadius: 6,
        padding: 10,
        fontSize: 14,
        backgroundColor: '#f9f9f9',
    },
    picker: {
        borderWidth: 1,
        borderColor: '#bdc3c7',
        borderRadius: 6,
        backgroundColor: '#f9f9f9',
    },
    button: {
        backgroundColor: '#3498db',
        padding: 12,
        borderRadius: 6,
        alignItems: 'center',
        marginTop: 8,
    },
    buttonDanger: {
        backgroundColor: '#e74c3c',
    },
    buttonSuccess: {
        backgroundColor: '#27ae60',
    },
    buttonText: {
        color: '#fff',
        fontWeight: '600',
        fontSize: 14,
    },
    userCard: {
        backgroundColor: '#f9f9f9',
        borderLeftWidth: 4,
        borderLeftColor: '#3498db',
        padding: 12,
        marginBottom: 8,
        borderRadius: 4,
    },
    userCardAdmin: {
        borderLeftColor: '#f39c12',
    },
    userEmail: {
        fontSize: 14,
        fontWeight: '600',
        color: '#2c3e50',
    },
    userRole: {
        fontSize: 12,
        color: '#7f8c8d',
        marginTop: 4,
    },
    userActions: {
        flexDirection: 'row',
        marginTop: 8,
        gap: 8,
    },
    smallButton: {
        flex: 1,
        paddingVertical: 6,
        paddingHorizontal: 8,
        borderRadius: 4,
        alignItems: 'center',
    },
    smallButtonText: {
        fontSize: 12,
        fontWeight: '500',
        color: '#fff',
    },
    logEntry: {
        backgroundColor: '#f9f9f9',
        padding: 10,
        marginBottom: 8,
        borderRadius: 4,
        borderLeftWidth: 3,
        borderLeftColor: '#95a5a6',
    },
    logAction: {
        fontSize: 13,
        fontWeight: '600',
        color: '#2c3e50',
    },
    logTime: {
        fontSize: 11,
        color: '#7f8c8d',
        marginTop: 4,
    },
    statsGrid: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 12,
    },
    statCard: {
        flex: 1,
        backgroundColor: '#ecf0f1',
        padding: 12,
        borderRadius: 6,
        marginHorizontal: 4,
        alignItems: 'center',
    },
    statNumber: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#2c3e50',
    },
    statLabel: {
        fontSize: 12,
        color: '#7f8c8d',
        marginTop: 4,
    },
    loading: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    errorText: {
        color: '#e74c3c',
        fontSize: 14,
        marginTop: 8,
    },
    successText: {
        color: '#27ae60',
        fontSize: 14,
        marginTop: 8,
    },
    tabContainer: {
        flexDirection: 'row',
        backgroundColor: '#ecf0f1',
        borderRadius: 6,
        padding: 4,
        marginBottom: 12,
    },
    tab: {
        flex: 1,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: 4,
        alignItems: 'center',
    },
    tabActive: {
        backgroundColor: '#3498db',
    },
    tabText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#7f8c8d',
    },
    tabTextActive: {
        color: '#fff',
    },
});

// ============================================
// COMPONENT
// ============================================

type Tab = 'assign' | 'users' | 'logs' | 'stats';

export const AdminManagement: React.FC = () => {
    const [activeTab, setActiveTab] = useState<Tab>('assign');
    const [email, setEmail] = useState('');
    const [selectedRole, setSelectedRole] = useState<AdminRole>('admin');
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [logs, setLogs] = useState<AdminLog[]>([]);
    const [stats, setStats] = useState<AdminStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);
    const [showUserModal, setShowUserModal] = useState(false);

    // ============================================
    // EFFECTS
    // ============================================

    useEffect(() => {
        if (activeTab === 'users') {
            loadUsers();
        } else if (activeTab === 'logs') {
            loadLogs();
        } else if (activeTab === 'stats') {
            loadStats();
        }
    }, [activeTab]);

    // ============================================
    // HANDLERS
    // ============================================

    const loadUsers = async () => {
        try {
            setLoading(true);
            setError('');
            const data = await getAllUsers();
            setUsers(data);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load users';
            setError(message);
            logger.error('Failed to load users:', err);
        } finally {
            setLoading(false);
        }
    };

    const loadLogs = async () => {
        try {
            setLoading(true);
            setError('');
            const { logs: data } = await getAdminLogs({ limit: 50 });
            setLogs(data);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load logs';
            setError(message);
            logger.error('Failed to load logs:', err);
        } finally {
            setLoading(false);
        }
    };

    const loadStats = async () => {
        try {
            setLoading(true);
            setError('');
            const data = await getAdminStats();
            setStats(data);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load stats';
            setError(message);
            logger.error('Failed to load stats:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleAssignAdmin = async () => {
        if (!email.trim()) {
            setError('Please enter an email address');
            return;
        }

        try {
            setLoading(true);
            setError('');
            await assignAdminRole(email, selectedRole);
            setSuccess(`Admin role '${selectedRole}' assigned to ${email}`);
            setEmail('');
            setTimeout(() => setSuccess(''), 3000);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to assign admin role';
            setError(message);
            logger.error('Failed to assign admin role:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleRevokeAdmin = async (userEmail: string) => {
        Alert.alert(
            'Revoke Admin Role',
            `Are you sure you want to revoke admin access from ${userEmail}?`,
            [
                { text: 'Cancel', onPress: () => {} },
                {
                    text: 'Revoke',
                    onPress: async () => {
                        try {
                            setLoading(true);
                            setError('');
                            await revokeAdminRole(userEmail);
                            setSuccess(`Admin role revoked from ${userEmail}`);
                            loadUsers();
                            setTimeout(() => setSuccess(''), 3000);
                        } catch (err) {
                            const message = err instanceof Error ? err.message : 'Failed to revoke admin role';
                            setError(message);
                            logger.error('Failed to revoke admin role:', err);
                        } finally {
                            setLoading(false);
                        }
                    },
                    style: 'destructive',
                },
            ]
        );
    };

    const handleViewUserDetails = async (user: AdminUser) => {
        try {
            setLoading(true);
            await getUserDetails(user.id);
            setSelectedUser(user);
            setShowUserModal(true);
        } catch (err) {
            logger.error('Failed to load user details:', err);
        } finally {
            setLoading(false);
        }
    };

    // ============================================
    // RENDER METHODS
    // ============================================

    const renderAssignTab = () => (
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Assign Admin Role</Text>

            <View style={styles.inputGroup}>
                <Text style={styles.label}>Email Address</Text>
                <TextInput
                    style={styles.input}
                    placeholder="user@example.com"
                    value={email}
                    onChangeText={setEmail}
                    keyboardType="email-address"
                    autoCapitalize="none"
                    editable={!loading}
                />
            </View>

            <View style={styles.inputGroup}>
                <Text style={styles.label}>Admin Role</Text>
                <View style={styles.picker}>
                    {ADMIN_ROLE_OPTIONS.map((option) => (
                        <TouchableOpacity
                            key={option.value}
                            style={[
                                styles.tabButton,
                                selectedRole === option.value && styles.tabButtonActive,
                            ]}
                            onPress={() => setSelectedRole(option.value)}
                            disabled={loading}
                        >
                            <Text
                                style={[
                                    styles.tabButtonText,
                                    selectedRole === option.value && styles.tabButtonTextActive,
                                ]}
                            >
                                {option.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </View>

            <TouchableOpacity
                style={[styles.button, styles.buttonSuccess]}
                onPress={handleAssignAdmin}
                disabled={loading}
            >
                {loading ? (
                    <ActivityIndicator color="#fff" />
                ) : (
                    <Text style={styles.buttonText}>Assign Admin Role</Text>
                )}
            </TouchableOpacity>

            {error && <Text style={styles.errorText}>{error}</Text>}
            {success && <Text style={styles.successText}>{success}</Text>}
        </View>
    );

    const renderUsersTab = () => (
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Users ({users.length})</Text>

            {loading ? (
                <View style={styles.loading}>
                    <ActivityIndicator size="large" color="#3498db" />
                </View>
            ) : (
                <FlatList
                    data={users}
                    keyExtractor={(item) => item.id}
                    scrollEnabled={false}
                    renderItem={({ item }) => (
                        <View
                            style={[
                                styles.userCard,
                                item.is_admin && styles.userCardAdmin,
                            ]}
                        >
                            <Text style={styles.userEmail}>{item.email}</Text>
                            <Text style={styles.userRole}>
                                {item.is_admin ? `Role: ${item.admin_role}` : 'Regular User'}
                            </Text>
                            <View style={styles.userActions}>
                                <TouchableOpacity
                                    style={[styles.smallButton, styles.button]}
                                    onPress={() => handleViewUserDetails(item)}
                                >
                                    <Text style={styles.smallButtonText}>View</Text>
                                </TouchableOpacity>
                                {item.is_admin && (
                                    <TouchableOpacity
                                        style={[styles.smallButton, styles.button, styles.buttonDanger]}
                                        onPress={() => handleRevokeAdmin(item.email)}
                                    >
                                        <Text style={styles.smallButtonText}>Revoke</Text>
                                    </TouchableOpacity>
                                )}
                            </View>
                        </View>
                    )}
                />
            )}

            {error && <Text style={styles.errorText}>{error}</Text>}
        </View>
    );

    const renderLogsTab = () => (
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Audit Logs</Text>

            {loading ? (
                <View style={styles.loading}>
                    <ActivityIndicator size="large" color="#3498db" />
                </View>
            ) : (
                <FlatList
                    data={logs}
                    keyExtractor={(item) => item.id}
                    scrollEnabled={false}
                    renderItem={({ item }) => (
                        <View style={styles.logEntry}>
                            <Text style={styles.logAction}>{item.action}</Text>
                            <Text style={styles.logTime}>
                                {new Date(item.created_at).toLocaleString()}
                            </Text>
                        </View>
                    )}
                />
            )}

            {error && <Text style={styles.errorText}>{error}</Text>}
        </View>
    );

    const renderStatsTab = () => (
        <View style={styles.section}>
            <Text style={styles.sectionTitle}>Dashboard Statistics</Text>

            {loading ? (
                <View style={styles.loading}>
                    <ActivityIndicator size="large" color="#3498db" />
                </View>
            ) : stats ? (
                <>
                    <View style={styles.statsGrid}>
                        <View style={styles.statCard}>
                            <Text style={styles.statNumber}>{stats.totalUsers}</Text>
                            <Text style={styles.statLabel}>Total Users</Text>
                        </View>
                        <View style={styles.statCard}>
                            <Text style={styles.statNumber}>{stats.totalAdmins}</Text>
                            <Text style={styles.statLabel}>Total Admins</Text>
                        </View>
                    </View>

                    <Text style={styles.sectionTitle}>Recent Actions</Text>
                    {Object.entries(stats.actionCounts).map(([action, count]) => (
                        <View key={action} style={styles.userCard}>
                            <Text style={styles.userEmail}>{action}</Text>
                            <Text style={styles.userRole}>{count} actions</Text>
                        </View>
                    ))}
                </>
            ) : null}

            {error && <Text style={styles.errorText}>{error}</Text>}
        </View>
    );

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.headerTitle}>Admin Management</Text>
            </View>

            <ScrollView>
                <View style={styles.tabContainer}>
                    {(['assign', 'users', 'logs', 'stats'] as Tab[]).map((tab) => (
                        <TouchableOpacity
                            key={tab}
                            style={[styles.tab, activeTab === tab && styles.tabActive]}
                            onPress={() => setActiveTab(tab)}
                        >
                            <Text
                                style={[
                                    styles.tabText,
                                    activeTab === tab && styles.tabTextActive,
                                ]}
                            >
                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>

                {activeTab === 'assign' && renderAssignTab()}
                {activeTab === 'users' && renderUsersTab()}
                {activeTab === 'logs' && renderLogsTab()}
                {activeTab === 'stats' && renderStatsTab()}
            </ScrollView>

            {/* User Details Modal */}
            <Modal
                visible={showUserModal}
                transparent
                animationType="slide"
                onRequestClose={() => setShowUserModal(false)}
            >
                <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.5)' }}>
                    <View
                        style={{
                            flex: 1,
                            backgroundColor: '#fff',
                            marginTop: 50,
                            borderTopLeftRadius: 16,
                            borderTopRightRadius: 16,
                            padding: 16,
                        }}
                    >
                        <TouchableOpacity onPress={() => setShowUserModal(false)}>
                            <Text style={{ fontSize: 16, fontWeight: '600', color: '#3498db' }}>
                                Close
                            </Text>
                        </TouchableOpacity>

                        {selectedUser && (
                            <ScrollView style={{ marginTop: 16 }}>
                                <Text style={styles.sectionTitle}>{selectedUser.email}</Text>
                                <View style={styles.userCard}>
                                    <Text style={styles.label}>Username</Text>
                                    <Text style={styles.userEmail}>{selectedUser.username}</Text>
                                </View>
                                <View style={styles.userCard}>
                                    <Text style={styles.label}>Admin Status</Text>
                                    <Text style={styles.userEmail}>
                                        {selectedUser.is_admin ? `${selectedUser.admin_role}` : 'Not Admin'}
                                    </Text>
                                </View>
                                <View style={styles.userCard}>
                                    <Text style={styles.label}>Created</Text>
                                    <Text style={styles.userEmail}>
                                        {new Date(selectedUser.created_at).toLocaleDateString()}
                                    </Text>
                                </View>
                            </ScrollView>
                        )}
                    </View>
                </View>
            </Modal>
        </View>
    );
};

export default AdminManagement;
