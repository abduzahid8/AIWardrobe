-- Service-only, resumable image evidence queue. No authoritative product writes.
begin;
set local lock_timeout='2s';
create table if not exists public.catalog_visual_worker_config(singleton boolean primary key default true check(singleton),enabled boolean not null default false);
alter table public.catalog_visual_worker_config enable row level security;
revoke all on public.catalog_visual_worker_config from public,anon,authenticated;
grant all on public.catalog_visual_worker_config to service_role;
insert into public.catalog_visual_worker_config(singleton,enabled) values(true,false) on conflict do nothing;
create table if not exists public.catalog_visual_jobs (
 product_id text primary key references public.shop_catalog(id) on delete cascade,
 source_hash text not null check(source_hash ~ '^[a-f0-9]{32}$'),
 priority integer not null default 10,
 state text not null default 'pending' check(state in ('pending','leased','succeeded','failed','skipped','unknown')),
 attempts integer not null default 0 check(attempts>=0),
 worker_id uuid, lease_until timestamptz, next_attempt_at timestamptz not null default now(),
 reason text check(reason in ('download_unavailable','inference_failed','no_confident_facets','source_changed','inactive','invalid_evidence')),
 updated_at timestamptz not null default now()
);
alter table public.catalog_visual_jobs enable row level security;
revoke all on public.catalog_visual_jobs from public,anon,authenticated;
grant all on public.catalog_visual_jobs to service_role;
create index if not exists catalog_visual_jobs_pending on public.catalog_visual_jobs(priority desc,next_attempt_at,product_id) where state in ('pending','failed','leased');

create or replace function public.catalog_visual_batch(p_action text,p_worker uuid,p_payload jsonb default '{}'::jsonb)
returns jsonb language plpgsql security invoker set search_path='' as $$
declare r jsonb; q public.catalog_visual_jobs%rowtype; s public.shop_catalog%rowtype;
 p jsonb; vector real[]; current_hash text; accepted integer:=0; skipped integer:=0; unknowns integer:=0; failures integer:=0;
 revision constant text:='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2';
begin
 if current_user<>'service_role' or p_worker is null then raise exception using errcode='42501',message='worker_auth_required';end if;
 if p_payload is null or jsonb_typeof(p_payload)<>'object' or octet_length(p_payload::text)>524288 then raise exception using errcode='22023',message='invalid_worker_payload';end if;
 if p_action<>'state' and not exists(select 1 from public.catalog_visual_worker_config where enabled) then raise exception using errcode='42501',message='worker_paused';end if;
 if p_action='seed' then
  insert into public.catalog_visual_jobs(product_id,source_hash,priority,state)
  select s.id,md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text),
   case when i.slot='shoes' then 1000 else 10 end +case when i.subtype in ('cardigan','blazer','dress-shirt','polo','chino','tailored-trouser','minimal-sneaker','loafer') then 100 else 0 end,
   case when v.product_id is not null and e.image_sha256 is not null then case when v.profile->'attributes'='{}'::jsonb then 'unknown' else 'succeeded' end else 'pending' end
  from public.shop_catalog s left join public.catalog_garment_intelligence i on i.product_id=s.id
  left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1 and v.model_revision=revision
   and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) and v.image_url=s.image_url and v.created_at>now()-interval '7 days'
   and (coalesce(i.slot,'other')<>'outfits' or v.profile#>>'{attributes,slot,value}'='outfits')
  left join public.catalog_image_embeddings e on e.image_sha256=v.image_sha256 and e.model_revision=v.model_revision
  where s.is_active
  on conflict(product_id) do update set source_hash=excluded.source_hash,priority=excluded.priority,
   state=case when public.catalog_visual_jobs.source_hash<>excluded.source_hash then excluded.state
     when excluded.state in ('succeeded','unknown') and public.catalog_visual_jobs.state<>'leased' then excluded.state
     when excluded.state='pending' and public.catalog_visual_jobs.state='unknown' and public.catalog_visual_jobs.updated_at<now()-interval '24 hours' then 'pending'
     when excluded.state='pending' and public.catalog_visual_jobs.state='succeeded' then 'pending'
     else public.catalog_visual_jobs.state end,
   worker_id=case when public.catalog_visual_jobs.source_hash<>excluded.source_hash then null else public.catalog_visual_jobs.worker_id end,
   lease_until=case when public.catalog_visual_jobs.source_hash<>excluded.source_hash then null else public.catalog_visual_jobs.lease_until end,
   attempts=case when public.catalog_visual_jobs.source_hash<>excluded.source_hash then 0 when excluded.state='pending' and (public.catalog_visual_jobs.state='succeeded' or (public.catalog_visual_jobs.state='unknown' and public.catalog_visual_jobs.updated_at<now()-interval '24 hours')) then 0 else public.catalog_visual_jobs.attempts end;
  return jsonb_build_object('seeded',true);
 elsif p_action='lease' then
  update public.catalog_visual_jobs set state='pending',worker_id=null,lease_until=null
   where state='leased' and lease_until<now();
  return (with picked as (select j.product_id from public.catalog_visual_jobs j join public.shop_catalog s on s.id=j.product_id
    where s.is_active and j.state in ('pending','failed') and j.attempts<3 and j.next_attempt_at<=now()
    order by j.priority desc,j.next_attempt_at,j.product_id for update of j skip locked limit 8), leased as (
    update public.catalog_visual_jobs j set state='leased',worker_id=p_worker,lease_until=now()+interval '20 minutes',attempts=j.attempts+1,updated_at=now()
    from picked where j.product_id=picked.product_id returning j.*)
   select coalesce(jsonb_agg(jsonb_build_object('id',s.id,'name',s.name,'description',s.description,'image_url',s.image_url,'source_hash',l.source_hash,'slot',coalesce(i.slot,'other')) order by l.priority desc),'[]')
   from leased l join public.shop_catalog s on s.id=l.product_id left join public.catalog_garment_intelligence i on i.product_id=s.id);
 elsif p_action='complete' then
  if jsonb_typeof(p_payload->'rows')<>'array' or jsonb_array_length(p_payload->'rows')>8 then raise exception using errcode='22023',message='invalid_worker_batch';end if;
  for r in select value from jsonb_array_elements(p_payload->'rows') loop
   select * into q from public.catalog_visual_jobs where product_id=r->>'product_id' and worker_id=p_worker and state='leased' for update;
   if not found then skipped:=skipped+1;continue;end if;
   select * into s from public.shop_catalog where id=q.product_id;
   if not found or not s.is_active then update public.catalog_visual_jobs set state='skipped',reason='inactive',worker_id=null,lease_until=null,updated_at=now() where product_id=q.product_id;skipped:=skipped+1;continue;end if;
   current_hash:=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text);
   if current_hash<>q.source_hash or r->>'source_hash' is distinct from q.source_hash then
    update public.catalog_visual_jobs set state='pending',source_hash=current_hash,attempts=0,reason='source_changed',worker_id=null,lease_until=null,updated_at=now() where product_id=q.product_id;skipped:=skipped+1;continue;
   end if;
   if r->>'status' in ('failed','unknown') and not(r?'profile') then
    update public.catalog_visual_jobs set state=case when attempts<3 then 'failed' else 'unknown' end,
      reason=case when r->>'reason'='inference_failed' then 'inference_failed' else 'download_unavailable' end,
      next_attempt_at=now()+interval '2 minutes'*attempts,worker_id=null,lease_until=null,updated_at=now() where product_id=q.product_id;
    failures:=failures+1;continue;
   end if;
   p:=r->'profile';
   if p->>'version' is distinct from '1' or p->>'revision' is distinct from revision or p->>'model' is distinct from 'google/siglip2-base-patch16-224'
    or p->>'sourceHash' is distinct from current_hash or p->>'imageUrl' is distinct from s.image_url
    or coalesce(p->>'imageSha256','') !~ '^[a-f0-9]{64}$' or jsonb_typeof(p->'attributes') is distinct from 'object'
    or octet_length(p::text)>=16000 or jsonb_typeof(r->'embedding') is distinct from 'array' then
     raise exception using errcode='22023',message='invalid_image_evidence';end if;
   if jsonb_array_length(r->'embedding')<>768 or exists(select 1 from jsonb_array_elements(r->'embedding') x where jsonb_typeof(x)<>'number') then raise exception using errcode='22023',message='invalid_embedding';end if;
   if exists(select 1 from jsonb_array_elements(r->'embedding') x where (x::text)::numeric not between -1 and 1)
    or (select sum(power((x::text)::numeric,2)) from jsonb_array_elements(r->'embedding') x) not between 0.9 and 1.1 then raise exception using errcode='22023',message='invalid_embedding';end if;
   if exists(select 1 from jsonb_each(p->'attributes') a where jsonb_typeof(value)<>'object'
      or jsonb_typeof(value->'confidence') is distinct from 'number' or jsonb_typeof(value->'margin') is distinct from 'number'
      or value->>'evidence' not in ('image-model','image-pixels','image-ocr')) then raise exception using errcode='22023',message='invalid_facet_provenance';end if;
   if exists(select 1 from jsonb_each(p->'attributes') a where (value->>'confidence')::numeric not between 0.8 and 1 or (value->>'margin')::numeric not between 0.1 and 1) then raise exception using errcode='22023',message='invalid_facet_confidence';end if;
   select array_agg((x::text)::real order by n) into vector from jsonb_array_elements(r->'embedding') with ordinality a(x,n);
   insert into public.catalog_image_embeddings values(p->>'imageSha256',revision,vector) on conflict(image_sha256,model_revision) do nothing;
   insert into public.catalog_visual_fashion(product_id,version,source_hash,image_url,image_sha256,model,model_revision,profile)
    values(q.product_id,1,current_hash,s.image_url,p->>'imageSha256',p->>'model',revision,p)
    on conflict(product_id,version) do update set source_hash=excluded.source_hash,image_url=excluded.image_url,image_sha256=excluded.image_sha256,model=excluded.model,model_revision=excluded.model_revision,profile=excluded.profile,created_at=now();
   update public.catalog_visual_jobs set state=case when p->'attributes'='{}'::jsonb then 'unknown' else 'succeeded' end,
    reason=case when p->'attributes'='{}'::jsonb then 'no_confident_facets' else null end,worker_id=null,lease_until=null,updated_at=now() where product_id=q.product_id;
   if p->'attributes'='{}'::jsonb then unknowns:=unknowns+1;else accepted:=accepted+1;end if;
  end loop;
  return jsonb_build_object('succeeded',accepted,'failed',failures,'skipped',skipped,'unknown',unknowns);
 elsif p_action='state' then
  return (select jsonb_build_object('total',count(*),'processed',count(*) filter(where state in ('succeeded','unknown','skipped')),
    'remaining',count(*) filter(where state in ('pending','leased','failed')),'succeeded',count(*) filter(where state='succeeded'),
    'failed',count(*) filter(where state='failed'),'skipped',count(*) filter(where state='skipped'),'unknown',count(*) filter(where state='unknown'),
    'leased',count(*) filter(where state='leased'),'profiles',(select count(*) from public.catalog_visual_fashion v join public.shop_catalog s on s.id=v.product_id where s.is_active and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)))
    from public.catalog_visual_jobs j join public.shop_catalog s on s.id=j.product_id where s.is_active);
 end if;
 raise exception using errcode='22023',message='invalid_worker_action';
end; $$;
revoke all on function public.catalog_visual_batch(text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.catalog_visual_batch(text,uuid,jsonb) to service_role;
notify pgrst,'reload schema';
commit;
