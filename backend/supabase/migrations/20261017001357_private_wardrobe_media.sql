insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
values ('wardrobe-images','wardrobe-images',false,10485760,
  array['image/jpeg','image/png','image/webp','image/heic','image/heif'])
on conflict(id) do nothing;

create policy wardrobe_media_owner_select on storage.objects for select to authenticated
using(bucket_id='wardrobe-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
create policy wardrobe_media_owner_insert on storage.objects for insert to authenticated
with check(bucket_id='wardrobe-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
create policy wardrobe_media_owner_update on storage.objects for update to authenticated
using(bucket_id='wardrobe-images' and (storage.foldername(name))[1]=(select auth.uid())::text)
with check(bucket_id='wardrobe-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
create policy wardrobe_media_owner_delete on storage.objects for delete to authenticated
using(bucket_id='wardrobe-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
