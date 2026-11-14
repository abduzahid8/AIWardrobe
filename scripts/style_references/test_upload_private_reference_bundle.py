"""Tests use synthetic JPEG bytes. No Supabase API calls or credentials."""
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

CODE = Path(__file__).with_name('upload_private_reference_bundle.py')
spec = importlib.util.spec_from_file_location('private_reference_uploader', CODE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def fixture(root, *, corrupt=False, approved=False):
    archive = root/'private.zip'
    images = []
    payloads = []
    for n in range(46):
        data = b'\xff\xd8\xff\xe0' + bytes([n]) * 6000
        sha = hashlib.sha256(data).hexdigest()
        path = f'images/{sha}.jpg'
        images.append({'id':f'rt_{n+1:03d}','sha256':sha,
                       'local_media_path':path,
                       'training_approved': approved and n == 4})
        payloads.append((path, data if not corrupt or n != 3 else data+b'oops'))
    meta = {'schema_version':2,'dataset_id':'image-teaste-2026-11-10',
            'media_count':46,'published_or_uploaded':False,'images':images}
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('manifest.json',json.dumps(meta))
        for path,data in payloads:
            z.writestr(path,data)
    return archive


class PrivateReferenceUploadTest(unittest.TestCase):
    def test_46_validated_without_network_or_key(self):
        with tempfile.TemporaryDirectory() as d:
            archive=fixture(Path(d))
            buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(app.main(['--archive',str(archive)]),0)
            result=json.loads(buffer.getvalue())
            self.assertEqual(result['photos'],46)
            self.assertEqual(result['uploaded'],0)
            self.assertFalse(result['model_trained'])

    def test_single_modified_image_blocks_entire_import(self):
        with tempfile.TemporaryDirectory() as d:
            archive=fixture(Path(d),corrupt=True)
            with self.assertRaisesRegex(ValueError,'media_sha_validation_failed'):
                app.validated_bundle(archive)

    def test_manifest_cannot_approve_model_training(self):
        with tempfile.TemporaryDirectory() as d:
            archive=fixture(Path(d),approved=True)
            with self.assertRaisesRegex(ValueError,'training_requires_separate_approval'):
                app.validated_bundle(archive)

    def test_upload_requires_explicit_confirmation_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            archive=fixture(Path(d))
            with self.assertRaises(SystemExit) as exit_info:
                app.main(['--archive',str(archive),'--upload'])
            self.assertNotEqual(exit_info.exception.code,0)

    def test_bucket_must_be_private_even_if_api_reports_success(self):
        class FakeStorage(app.Storage):
            def __init__(self):
                self.calls=[]
            def request(self,method,resource,data=None,content_type='application/json'):
                self.calls.append((method,resource))
                return (200,json.dumps({'public':True,'id':'stylist6-private-research'}).encode())
        client=FakeStorage()
        with self.assertRaisesRegex(ValueError,'bucket_is_not_proven_private'):
            client.ensure_private_bucket('stylist6-private-research',False)

    def test_real_credential_missing_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'invalid_supabase_url'):
            app.Storage('','')
        with self.assertRaisesRegex(ValueError,'missing_service_role_key'):
            app.Storage('https://test-project.supabase.co','')


if __name__=='__main__':
    unittest.main()
