-- Guide page content table for editable onboarding/guide page
CREATE TABLE IF NOT EXISTS guide_page (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL DEFAULT 'Our Online Boutique AI',
  subtitle TEXT NOT NULL DEFAULT 'A personalised journey into the style of Brunello Cucinelli',
  cta_text TEXT NOT NULL DEFAULT 'DISCOVER THE NEW WEBSITE',
  cta_url TEXT,
  hero_image_url TEXT,
  background_color TEXT DEFAULT '#F5F5F5',
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE guide_page ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read active guide content
CREATE POLICY "Anyone can read active guide content"
  ON guide_page FOR SELECT
  USING (is_active = true);

-- Allow service_role to manage guide content
CREATE POLICY "Service role can manage guide content"
  ON guide_page FOR ALL
  USING (auth.role() = 'service_role');

-- Allow admin users to manage guide content
CREATE POLICY "Admin users can manage guide content"
  ON guide_page FOR ALL
  USING (
    auth.uid() IN (
      SELECT id FROM profiles
      WHERE is_admin = true
      OR email = 'info@aiwardrobe.club'
    )
  );

-- Insert default content
INSERT INTO guide_page (title, subtitle, cta_text, cta_url, hero_image_url, background_color)
VALUES (
  'Our Online Boutique AI',
  'A personalised journey into the style of Brunello Cucinelli',
  'DISCOVER THE NEW WEBSITE',
  'https://aiwardrobe.club',
  NULL,
  '#F5F5F5'
)
ON CONFLICT DO NOTHING;

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_guide_page_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER guide_page_updated_at
  BEFORE UPDATE ON guide_page
  FOR EACH ROW
  EXECUTE FUNCTION update_guide_page_updated_at();
