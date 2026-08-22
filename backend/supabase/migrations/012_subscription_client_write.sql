-- 012_subscription_client_write.sql
--
-- Allow authenticated users to INSERT and UPDATE their own rows in the
-- subscriptions table.  Previously only service_role (webhook) could write,
-- which meant a client-side IAP purchase that succeeded through RevenueCat
-- but whose webhook hadn't fired yet left the user with no subscription row.
-- verifySubscriptionFromServer() checks subscriptions FIRST and falls back
-- to profiles — without a row here the user appears "free" until the
-- webhook arrives (which may be delayed or lost).

-- Users can insert their own subscription rows
DROP POLICY IF EXISTS "Users can insert own subscriptions" ON public.subscriptions;
CREATE POLICY "Users can insert own subscriptions"
    ON public.subscriptions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can update their own subscription rows (e.g. tier upgrade)
DROP POLICY IF EXISTS "Users can update own subscriptions" ON public.subscriptions;
CREATE POLICY "Users can update own subscriptions"
    ON public.subscriptions FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
