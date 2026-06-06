# Manual Admin Setup - Direct Database Method

If you've lost access to your previous admin Gmail, you can manually add a new admin directly in Supabase.

## Quick Steps

### 1. Go to Supabase Dashboard
- Open https://app.supabase.com
- Select your AIWardrobe project
- Go to **SQL Editor**

### 2. Run This SQL Command

Replace `your-new-email@gmail.com` with your new Gmail address:

```sql
UPDATE profiles 
SET is_admin = true, admin_role = 'super_admin' 
WHERE email = 'your-new-email@gmail.com';
```

**Example:**
```sql
UPDATE profiles 
SET is_admin = true, admin_role = 'super_admin' 
WHERE email = 'newemail@gmail.com';
```

### 3. Verify It Worked

Run this query to check:

```sql
SELECT id, email, is_admin, admin_role FROM profiles WHERE email = 'your-new-email@gmail.com';
```

You should see:
- `is_admin`: true
- `admin_role`: super_admin

## If User Doesn't Exist Yet

If the user hasn't signed up yet, you need to:

1. **Create the user in Supabase Auth first**
   - Go to **Authentication** → **Users**
   - Click **Add user**
   - Enter email and password
   - Click **Create user**

2. **Then run the SQL command above** to make them admin

## Alternative: Using Supabase Dashboard UI

If you prefer the UI instead of SQL:

1. Go to **SQL Editor** in Supabase
2. Click **New Query**
3. Paste the SQL command
4. Click **Run**

## Verify Admin Access

After setting up, the user should:
1. Sign in with their Gmail
2. See the Admin Panel in the app
3. Be able to manage other admins

## Troubleshooting

### "No rows updated" error
- The email doesn't exist in the profiles table
- Make sure the user has signed up first
- Check the exact email spelling

### Can't see Admin Panel after login
- Log out and log back in
- Clear app cache
- Restart the app

### Need to remove admin access?
```sql
UPDATE profiles 
SET is_admin = false, admin_role = null 
WHERE email = 'email@gmail.com';
```

## Admin Roles

Choose one when setting up:

```sql
-- Super Admin (full access)
UPDATE profiles 
SET is_admin = true, admin_role = 'super_admin' 
WHERE email = 'email@gmail.com';

-- Admin (can manage content)
UPDATE profiles 
SET is_admin = true, admin_role = 'admin' 
WHERE email = 'email@gmail.com';

-- Moderator (limited access)
UPDATE profiles 
SET is_admin = true, admin_role = 'moderator' 
WHERE email = 'email@gmail.com';
```

## Need Help?

If the user still doesn't have admin access:

1. Check the email is spelled correctly
2. Verify the user has signed up
3. Check the `profiles` table directly:
   ```sql
   SELECT * FROM profiles WHERE email = 'your-email@gmail.com';
   ```
4. Make sure `is_admin` is `true` and `admin_role` is set

---

**That's it!** Your new Gmail is now an admin. You can log in and start managing the app.
