# Admin Role Assignment System

Complete guide for managing admin roles and permissions in AIWardrobe.

## Overview

The admin role assignment system provides a secure, audited way to grant administrative access to users. It supports multiple admin roles with different permission levels and maintains a complete audit trail of all admin actions.

## Features

- **Multiple Admin Roles**: Super Admin, Admin, Moderator
- **Audit Logging**: Complete trail of all admin actions
- **Fine-grained Permissions**: Grant specific permissions to admins
- **User Management**: View and manage all users
- **Dashboard Statistics**: Monitor admin activity
- **Batch Operations**: Assign/revoke roles for multiple users

## Admin Roles

### Super Admin
- Full system access
- Can assign/revoke admin roles
- Can manage permissions
- Can view all audit logs
- Can delete users
- Can access all admin features

### Admin
- Can view users and their details
- Can view audit logs
- Can access admin dashboard
- Cannot assign/revoke other admins
- Cannot manage permissions

### Moderator
- Limited access
- Can view users
- Can view audit logs
- Cannot modify user data
- Cannot assign/revoke roles

## Database Schema

### New Tables

#### `profiles` (Extended)
```sql
ALTER TABLE profiles ADD COLUMN:
- is_admin BOOLEAN DEFAULT FALSE
- admin_role TEXT ('super_admin', 'admin', 'moderator')
- admin_assigned_at TIMESTAMP
- admin_assigned_by UUID (references profiles.id)
```

#### `admin_logs`
Audit trail for all admin actions:
```sql
CREATE TABLE admin_logs (
    id UUID PRIMARY KEY,
    admin_id UUID REFERENCES profiles(id),
    action TEXT ('assign_admin', 'revoke_admin', 'update_role', 'delete_user', 'view_user_data'),
    target_user_id UUID REFERENCES profiles(id),
    details JSONB,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP
)
```

#### `admin_permissions`
Fine-grained permission management:
```sql
CREATE TABLE admin_permissions (
    id UUID PRIMARY KEY,
    admin_id UUID REFERENCES profiles(id),
    permission TEXT,
    granted_at TIMESTAMP,
    granted_by UUID REFERENCES profiles(id)
)
```

## API Endpoints

All endpoints require authentication. Super Admin endpoints require `admin_role = 'super_admin'`.

### User Management

#### GET `/api/admin/users`
List all users with admin status.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "email": "user@example.com",
      "username": "username",
      "is_admin": false,
      "admin_role": null,
      "admin_assigned_at": null,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 100
}
```

#### GET `/api/admin/users/:userId`
Get detailed information about a specific user.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "email": "user@example.com",
    "username": "username",
    "gender": "male",
    "profile_image": "url",
    "is_admin": true,
    "admin_role": "admin",
    "admin_assigned_at": "2024-01-01T00:00:00Z",
    "admin_assigned_by": "uuid",
    "subscription_tier": "premium",
    "subscription_expires_at": "2024-12-31T00:00:00Z",
    "is_active": true,
    "is_email_verified": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```

### Admin Role Management

#### POST `/api/admin/assign-admin`
Assign admin role to a user. **Requires Super Admin**.

**Request:**
```json
{
  "email": "user@example.com",
  "role": "admin"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Admin role 'admin' assigned to user@example.com",
  "data": {
    "userId": "uuid",
    "email": "user@example.com",
    "role": "admin"
  }
}
```

#### POST `/api/admin/revoke-admin`
Revoke admin role from a user. **Requires Super Admin**.

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Admin role revoked from user@example.com",
  "data": {
    "userId": "uuid",
    "email": "user@example.com"
  }
}
```

### Audit Logs

#### GET `/api/admin/logs`
Get admin action logs with pagination and filtering.

**Query Parameters:**
- `limit` (default: 50) - Number of logs to return
- `offset` (default: 0) - Pagination offset
- `admin_id` - Filter by admin who performed the action
- `target_user_id` - Filter by target user
- `action` - Filter by action type

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "admin_id": "uuid",
      "action": "assign_admin",
      "target_user_id": "uuid",
      "details": { "role": "admin", "previous_role": null },
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 100
  }
}
```

### Permissions

#### GET `/api/admin/permissions/:userId`
Get permissions for an admin user.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid",
      "admin_id": "uuid",
      "permission": "manage_users",
      "granted_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### POST `/api/admin/permissions`
Grant a permission to an admin. **Requires Super Admin**.

**Request:**
```json
{
  "admin_id": "uuid",
  "permission": "manage_users"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Permission 'manage_users' granted",
  "data": {
    "admin_id": "uuid",
    "permission": "manage_users"
  }
}
```

#### DELETE `/api/admin/permissions/:permissionId`
Revoke a permission from an admin. **Requires Super Admin**.

**Response:**
```json
{
  "success": true,
  "message": "Permission revoked"
}
```

### Statistics

#### GET `/api/admin/stats`
Get admin dashboard statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "totalUsers": 1000,
    "totalAdmins": 5,
    "actionCounts": {
      "assign_admin": 10,
      "revoke_admin": 2,
      "view_user_data": 150
    }
  }
}
```

## Client-Side Service

The `adminService.ts` provides TypeScript functions for all admin operations:

```typescript
import {
  getAllUsers,
  getUserDetails,
  assignAdminRole,
  revokeAdminRole,
  getAdminLogs,
  getAdminPermissions,
  grantPermission,
  revokePermission,
  getAdminStats,
  assignAdminRoleBatch,
  revokeAdminRoleBatch,
} from '../services/adminService';

// Assign admin role
await assignAdminRole('user@example.com', 'admin');

// Get all users
const users = await getAllUsers();

// Get audit logs
const { logs, pagination } = await getAdminLogs({ limit: 50, offset: 0 });

// Batch operations
const results = await assignAdminRoleBatch(
  ['user1@example.com', 'user2@example.com'],
  'admin'
);
```

## UI Component

The `AdminManagement.tsx` component provides a complete UI for admin management:

```typescript
import AdminManagement from '../components/AdminManagement';

// Use in your app
<AdminManagement />
```

### Features:
- **Assign Tab**: Assign admin roles to users by email
- **Users Tab**: View all users and their admin status
- **Logs Tab**: View audit trail of admin actions
- **Stats Tab**: Dashboard statistics

## Setup Instructions

### 1. Run Database Migration

Apply the migration to your Supabase database:

```bash
# Using Supabase CLI
supabase migration up

# Or manually run the SQL from supabase/migrations/add_admin_management.sql
```

### 2. Update API

The admin routes are already mounted in `/api/index.js`:

```javascript
import adminRoutes from "./routes/admin.js";
app.use("/api/admin", authenticateToken, adminRoutes);
```

### 3. Add to Your App

Import and use the AdminManagement component:

```typescript
import AdminManagement from './src/components/AdminManagement';

// In your navigation or admin screen
<AdminManagement />
```

## Usage Examples

### Assign Admin Role via API

```bash
curl -X POST http://localhost:3000/api/admin/assign-admin \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newadmin@example.com",
    "role": "admin"
  }'
```

### Assign Admin Role via Client

```typescript
import { assignAdminRole } from '../services/adminService';

try {
  await assignAdminRole('newadmin@example.com', 'admin');
  console.log('Admin role assigned successfully');
} catch (error) {
  console.error('Failed to assign admin role:', error);
}
```

### Get All Admins

```typescript
import { getAllUsers } from '../services/adminService';

const users = await getAllUsers();
const admins = users.filter(u => u.is_admin);
console.log('Total admins:', admins.length);
```

### View Audit Logs

```typescript
import { getAdminLogs } from '../services/adminService';

const { logs, pagination } = await getAdminLogs({
  limit: 100,
  action: 'assign_admin'
});

logs.forEach(log => {
  console.log(`${log.action} by ${log.admin_id} at ${log.created_at}`);
});
```

## Security Considerations

1. **Authentication**: All endpoints require valid Supabase JWT token
2. **Authorization**: Super Admin role required for sensitive operations
3. **Audit Trail**: All admin actions are logged with IP and user agent
4. **RLS Policies**: Database-level row security prevents unauthorized access
5. **Rate Limiting**: Admin endpoints are subject to rate limiting
6. **Input Validation**: All inputs are validated server-side

## Troubleshooting

### "Super admin access required" error
- Verify the user has `admin_role = 'super_admin'` in the database
- Check that the user's token is valid and not expired

### "User not found" error
- Verify the email address is correct
- Check that the user exists in the profiles table

### Audit logs not appearing
- Verify the admin_logs table exists
- Check that the user has admin privileges
- Review server logs for errors

## Best Practices

1. **Limit Super Admins**: Only grant super admin role to trusted users
2. **Regular Audits**: Review audit logs regularly for suspicious activity
3. **Batch Operations**: Use batch functions for multiple user updates
4. **Permissions**: Use fine-grained permissions for specific admin tasks
5. **Monitoring**: Set up alerts for admin role assignments
6. **Documentation**: Keep records of who has admin access and why

## Future Enhancements

- [ ] Two-factor authentication for admin accounts
- [ ] Admin session management and timeout
- [ ] Advanced filtering and search in audit logs
- [ ] Admin activity reports and analytics
- [ ] Scheduled admin role expiration
- [ ] Admin action approval workflow
- [ ] Integration with external identity providers
