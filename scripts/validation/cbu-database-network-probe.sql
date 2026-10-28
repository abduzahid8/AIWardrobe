begin;
set local lock_timeout = '2s';
set local statement_timeout = '15s';
create schema readiness_cbu_network_probe;
revoke all on schema readiness_cbu_network_probe from public, anon, authenticated;
create extension http with schema readiness_cbu_network_probe;
select readiness_cbu_network_probe.http_set_curlopt('CURLOPT_CONNECTTIMEOUT_MS','4000');
select readiness_cbu_network_probe.http_set_curlopt('CURLOPT_TIMEOUT_MS','8000');
select readiness_cbu_network_probe.http_set_curlopt('CURLOPT_SSL_VERIFYHOST','2');
select readiness_cbu_network_probe.http_set_curlopt('CURLOPT_SSL_VERIFYPEER','1');
create temporary table readiness_cbu_network_result(target text,status integer,content_type text,body_bytes integer,array_rows integer,error_code text,error_message text,elapsed_ms integer);
do $probe$
declare response record; started timestamptz; target text;
begin
for target in select unnest(array['https://www.google.com/','https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/2026-10-25/']) loop
started := clock_timestamp();
begin
  select * into response from readiness_cbu_network_probe.http_get(target);
  insert into readiness_cbu_network_result values(target,response.status,response.content_type,octet_length(response.content),case when response.status=200 and response.content_type like '%json%' then jsonb_array_length(response.content::jsonb) else null end,null,null,extract(epoch from clock_timestamp()-started)*1000);
exception when others then
  insert into readiness_cbu_network_result(target,error_code,error_message,elapsed_ms) values(target,sqlstate,left(sqlerrm,400),extract(epoch from clock_timestamp()-started)*1000);
end;
end loop;
end;
$probe$;
select * from readiness_cbu_network_result;
rollback;
