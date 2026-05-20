-- Create table for home screen occasions and styles (editable via Supabase / Admin panel)
CREATE TABLE IF NOT EXISTS public.home_occasions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  occasion TEXT NOT NULL,          -- e.g., 'Team Collaboration' or 'Night-Time Dinner' or localization keys
  style TEXT NOT NULL,             -- e.g., 'business_casual' or 'old_money'
  is_active BOOLEAN NOT NULL DEFAULT true,
  sort_order INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.home_occasions ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read active occasions
CREATE POLICY home_occasions_public_read
  ON public.home_occasions FOR SELECT
  USING (is_active = true);

-- Allow admin users and service_role to manage occasions
CREATE POLICY home_occasions_admin_write
  ON public.home_occasions FOR ALL
  USING (
    auth.role() = 'service_role'
    OR public.is_admin()
  )
  WITH CHECK (
    auth.role() = 'service_role'
    OR public.is_admin()
  );

-- Insert default occasions
INSERT INTO public.home_occasions (occasion, style, sort_order)
VALUES 
  ('Team Collaboration', 'business_casual', 10),
  ('Night-Time Dinner', 'old_money', 20)
ON CONFLICT DO NOTHING;

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_home_occasions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER home_occasions_updated_at
  BEFORE UPDATE ON public.home_occasions
  FOR EACH ROW
  EXECUTE FUNCTION update_home_occasions_updated_at();
