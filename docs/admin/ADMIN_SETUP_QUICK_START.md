# Admin Role Assignment - Quick Start Guide

Get admin role assignment functionality up and running in 5 minutes.

## What You Get

✅ Assign admin roles to new Gmail accounts  
✅ Revoke admin access  
✅ View all users and their admin status  
✅ Complete audit trail of admin actions  
✅ Dashboard with statistics  
✅ Fine-grained permission management  

## Quick Setup

### Step 1: Apply Database Migration

Run the migration to create admin tables:

```bash
# Option A: Using Supabase CLI
supabase migration up

# Option B: Manually in Supabase SQL Editor
# Copy and paste the contents of:
# supabase/migrations/add_admin_management.sql
```

### Step 2: Verify API Routes

The admin routes are already added to `/api/index.js`. Verify they're mounted:

```javascript
// Should see this in api/index.js
import adminRoutes from "./routes/admin.js";
app.use("/api/admin", authenticateToken, adminRoutes);
```

### Step 3: Add Admin UI to Your App

Import the AdminManagement component in your admin screen:

```typescript
import AdminManagement from './src/components/AdminManagement';

export function AdminScreen() {
  return <AdminManagement />;
}
```

## Assign Admin Role to New Gmail Account

### Via UI (Easiest)

1. Open Admin Management screen
2. Click "Assign" tab
3. Enter Gmail address: `user@gmail.com`
4. Select role: Admin, Moderator, or Super Admin
5. Click "Assign Admin Role"

### Via API (Programmatic)

```bash
curl -X POST http://localhost:3000/api/admin/assign-admin \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@gmail.com",
    "role": "admin"
  }'
```

### Via TypeScript

```typescript
import { assignAdminRole } from './src/services/adminService';

await assignAdminRole('user@gmail.com', 'admin');
```

## Admin Roles Explained

| Role | Can Assign Admins | Can View Users | Can View Logs | Can Manage Permissions |
|------|-------------------|----------------|---------------|----------------------|
| Super Admin | ✅ | ✅ | ✅ | ✅ |
| Admin | ❌ | ✅ | ✅ | ❌ |
| Moderator | ❌ | ✅ | ✅ | ❌ |

## Common Tasks

### View All Admins

```typescript
import { getAllUsers } from './src/services/adminService';

const users = await getAllUsers();
const admins = users.filter(u => u.is_admin);
console.log('Admins:', admins);
```

### Revoke Admin Access

```typescript
import { revokeAdminRole } from './src/services/adminService';

await revokeAdminRole('user@gmail.com');
```

### View Audit Logs

```typescript
import { getAdminLogs } from './src/services/adminService';

const { logs } = await getAdminLogs({ limit: 50 });
logs.forEach(log => {
  console.log(`${log.action} at ${log.created_at}`);
});
```

### Batch Assign Admins

```typescript
import { assignAdminRoleBatch } from './src/services/adminService';

const emails = [
  'admin1@gmail.com',
  'admin2@gmail.com',
  'admin3@gmail.com'
];

const results = await assignAdminRoleBatch(emails, 'admin');
console.log('Success:', results.success);
console.log('Failed:', results.failed);
```

## Files Created

| File | Purpose |
|------|---------|
| `supabase/migrations/add_admin_management.sql` | Database schema |
| `api/routes/admin.js` | Backend API endpoints |
| `src/services/adminService.ts` | Client-side service |
| `src/components/AdminManagement.tsx` | Admin UI component |
| `ADMIN_MANAGEMENT.md` | Full documentation |

## API Endpoints

All endpoints require authentication and are prefixed with `/api/admin`:

```
POST   /assign-admin          - Assign admin role (Super Admin only)
POST   /revoke-admin          - Revoke admin role (Super Admin only)
GET    /users                 - List all users
GET    /users/:userId         - Get user details
GET    /logs                  - Get audit logs
GET    /permissions/:userId   - Get user permissions
POST   /permissions           - Grant permission (Super Admin only)
DELETE /permissions/:id       - Revoke permission (Super Admin only)
GET    /stats                 - Get dashboard stats
```

## Testing

### Test Assign Admin

```bash
# 1. Get your auth token from Supabase
TOKEN="your_supabase_token"

# 2. Assign admin role
curl -X POST http://localhost:3000/api/admin/assign-admin \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@gmail.com",
    "role": "admin"
  }'

# 3. Verify in Supabase dashboard
# SELECT * FROM profiles WHERE email = 'test@gmail.com';
# Should show: is_admin = true, admin_role = 'admin'
```

### Test Audit Logs

```bash
curl -X GET http://localhost:3000/api/admin/logs \
  -H "Authorization: Bearer $TOKEN"
```

## Troubleshooting

### "Super admin access required"
- Your user must have `admin_role = 'super_admin'`
- Set it manually in Supabase: 
  ```sql
  UPDATE profiles 
  SET is_admin = true, admin_role = 'super_admin' 
  WHERE email = 'your@email.com';
  ```

### "User not found"
- Verify the email exists in the profiles table
- Check spelling and case sensitivity

### API returns 401 Unauthorized
- Verify your auth token is valid
- Check token hasn't expired
- Ensure Authorization header format: `Bearer <token>`

## Next Steps

1. ✅ Set up the first Super Admin (yourself)
2. ✅ Assign admin roles to team members
3. ✅ Monitor audit logs regularly
4. ✅ Review ADMIN_MANAGEMENT.md for advanced features

## Support

For detailed documentation, see: `ADMIN_MANAGEMENT.md`

For API reference, see: `api/routes/admin.js`

For client service, see: `src/services/adminService.ts`
