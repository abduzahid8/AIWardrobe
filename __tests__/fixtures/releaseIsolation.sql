-- Disposable synthetic DB only. Probe actual grants and RLS as non-owner roles.
do $$ declare t text; owner_col text; begin
 for t,owner_col in
 select c.table_name, c.column_name from information_schema.columns c
 where c.table_schema='public' and c.column_name in ('user_id','owner_id','inviter_user_id')
 loop
  if not (select relrowsecurity from pg_class where oid=('public.'||t)::regclass) then
   raise exception 'RLS missing: %',t;
  end if;
  if not has_table_privilege('authenticated','public.'||t,'SELECT') then raise exception 'owner grant missing: %',t; end if;
  if has_table_privilege('anon','public.'||t,'SELECT') then raise exception 'anonymous grant: %',t; end if;
 end loop;
end $$;
insert into public.style_evidence(user_id,source_type,entity_type,entity_key)
values ('11111111-1111-1111-1111-111111111111','manual_feedback','garment','synthetic');
set role authenticated;
select set_config('request.jwt.claim.sub','22222222-2222-2222-2222-222222222222',false);
do $$ begin
 if exists(select 1 from public.style_evidence) then raise exception 'cross-user SELECT leaked'; end if;
 begin
  insert into public.style_evidence(user_id,source_type,entity_type,entity_key)
  values ('11111111-1111-1111-1111-111111111111','manual_feedback','garment','forged');
  raise exception 'cross-user INSERT accepted';
 exception when insufficient_privilege then null;
 end;
end $$;
select set_config('request.jwt.claim.sub','11111111-1111-1111-1111-111111111111',false);
do $$ begin
 if (select count(*) from public.style_evidence) <> 1 then raise exception 'owner read denied'; end if;
 begin
  update public.style_evidence set user_id='22222222-2222-2222-2222-222222222222';
  raise exception 'owner reassignment accepted';
 exception when insufficient_privilege then null;
 end;
end $$;
reset role;

-- Server-owned outcomes/rewards cannot be forged by clients.
do $$ begin
 if has_table_privilege('authenticated','public.growth_referrals','INSERT') or
    has_table_privilege('authenticated','public.growth_savings_ledger','INSERT') then
   raise exception 'client can forge rewards/savings';
 end if;
 if to_regclass('public.style_evidence_user_key_idx') is null or
    to_regclass('public.decision_events_user_feature_idx') is null then raise exception 'index missing'; end if;
end $$;
insert into public.visual_assets(id,user_id,kind) values
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa','11111111-1111-1111-1111-111111111111','original_upload'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb','22222222-2222-2222-2222-222222222222','original_upload');
set role authenticated;
select set_config('request.jwt.claim.sub','11111111-1111-1111-1111-111111111111',false);
insert into public.reference_looks(user_id,asset_id) values ('11111111-1111-1111-1111-111111111111','aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa');
do $$ begin
 begin
  insert into public.reference_looks(user_id,asset_id) values ('11111111-1111-1111-1111-111111111111','bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');
  raise exception 'cross-owner reference accepted';
 exception when foreign_key_violation then null; end;
 begin
  insert into public.visual_assets(user_id,kind,original_asset_id) values ('11111111-1111-1111-1111-111111111111','thumbnail','bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');
  raise exception 'cross-owner original accepted';
 exception when foreign_key_violation then null; end;
 begin
  insert into public.visual_evidence(user_id,source_asset_id,entity_type,attribute_type,source) values ('11111111-1111-1111-1111-111111111111','bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb','wardrobe_item','color','user');
  raise exception 'cross-owner evidence accepted';
 exception when foreign_key_violation then null; end;
 begin
  insert into public.tryon_requests(user_id,person_asset_id,garment_variant_id,category) values ('11111111-1111-1111-1111-111111111111','bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb','synthetic','shirt');
  raise exception 'cross-owner tryon accepted';
 exception when foreign_key_violation then null; end;
end $$;
reset role;

-- Owner-qualified references also reject cross-owner updates/supersession.
insert into public.visual_evidence(id,user_id,source_asset_id,entity_type,attribute_type,source)
values ('cccccccc-cccc-cccc-cccc-cccccccccccc','11111111-1111-1111-1111-111111111111','aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa','wardrobe_item','color','user'),
('dddddddd-dddd-dddd-dddd-dddddddddddd','22222222-2222-2222-2222-222222222222','bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb','wardrobe_item','color','user');
set role authenticated;
select set_config('request.jwt.claim.sub','11111111-1111-1111-1111-111111111111',false);
do $$ begin
 begin
  update public.reference_looks set asset_id='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb';
  raise exception 'cross-owner reference update accepted';
 exception when foreign_key_violation then null; end;
 begin
  update public.visual_evidence set superseded_by='dddddddd-dddd-dddd-dddd-dddddddddddd';
  raise exception 'cross-owner supersession accepted';
 exception when foreign_key_violation then null; end;
end $$;
reset role;
-- Removing an owned original must clear references without clearing row ownership.
delete from public.visual_assets where id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
do $$ begin
 if not exists(select 1 from public.reference_looks where user_id='11111111-1111-1111-1111-111111111111' and asset_id is null)
 then raise exception 'owned asset deletion did not retain reference owner'; end if;
 if exists(select 1 from public.visual_evidence where source_asset_id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
 then raise exception 'owned asset deletion did not cascade evidence'; end if;
end $$;
