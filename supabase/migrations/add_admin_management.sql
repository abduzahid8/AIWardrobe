-- ============================================
-- ADMIN MANAGEMENT SYSTEM
-- ============================================

-- Add admin role column to profiles if not exists
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS admin_role TEXT CHECK (admin_role IN ('super_admin', 'admin', 'moderator')) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS admin_assigned_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS admin_assigned_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL;

-- Create admin_logs table for audit trail
CREATE TABLE IF NOT EXISTS public.admin_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admin_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK (action IN ('assign_admin', 'revoke_admin', 'update_role', 'delete_user', 'view_user_data')),
    target_user_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    details JSONB DEFAULT '{}',
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create admin_permissions table for fine-grained access control
CREATE TABLE IF NOT EXISTS public.admin_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admin_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    permission TEXT NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    UNIQUE(admin_id, permission)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_profiles_is_admin ON public.profiles(is_admin);
CREATE INDEX IF NOT EXISTS idx_profiles_admin_role ON public.profiles(admin_role);
CREATE INDEX IF NOT EXISTS idx_admin_logs_admin_id ON public.admin_logs(admin_id);
CREATE INDEX IF NOT EXISTS idx_admin_logs_target_user_id ON public.admin_logs(target_user_id);
CREATE INDEX IF NOT EXISTS idx_admin_logs_created_at ON public.admin_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_admin_permissions_admin_id ON public.admin_permissions(admin_id);

-- Create SQL function to check if user is admin
CREATE OR REPLACE FUNCTION public.is_admin(user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.profiles
        WHERE id = user_id AND is_admin = TRUE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create SQL function to get admin role
CREATE OR REPLACE FUNCTION public.get_admin_role(user_id UUID)
RETURNS TEXT AS $$
DECLARE
    role TEXT;
BEGIN
    SELECT admin_role INTO role FROM public.profiles
    WHERE id = user_id AND is_admin = TRUE;
    RETURN role;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create SQL function to log admin actions
CREATE OR REPLACE FUNCTION public.log_admin_action(
    p_admin_id UUID,
    p_action TEXT,
    p_target_user_id UUID DEFAULT NULL,
    p_details JSONB DEFAULT '{}'::JSONB,
    p_ip_address TEXT DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    log_id UUID;
BEGIN
    INSERT INTO public.admin_logs (admin_id, action, target_user_id, details, ip_address, user_agent)
    VALUES (p_admin_id, p_action, p_target_user_id, p_details, p_ip_address, p_user_agent)
    RETURNING id INTO log_id;
    RETURN log_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- RLS Policies for admin_logs
ALTER TABLE public.admin_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admins can view admin logs" ON public.admin_logs
    FOR SELECT USING (
        public.is_admin(auth.uid())
    );

CREATE POLICY "Admins can insert admin logs" ON public.admin_logs
    FOR INSERT WITH CHECK (
        public.is_admin(auth.uid()) AND admin_id = auth.uid()
    );

-- RLS Policies for admin_permissions
ALTER TABLE public.admin_permissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admins can view admin permissions" ON public.admin_permissions
    FOR SELECT USING (
        public.is_admin(auth.uid())
    );

CREATE POLICY "Super admins can manage permissions" ON public.admin_permissions
    FOR ALL USING (
        public.get_admin_role(auth.uid()) = 'super_admin'
    );
