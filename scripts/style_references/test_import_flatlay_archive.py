"""Synthetic input tests for the 46-source private archive importer."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('flatlay',Path(__file__).with_name('import_flatlay_archive.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PrivateFlatlayImportTests(unittest.TestCase):
    def fixture(self,root, missing=False, duplicate=False, bad_member=False):
        rows=[]
        for n in range(1,47):
            rows.append({'id':f'rt_{n:03d}', 'filename':f'photo_{n}_2026-11-10_09-23-10.jpg',
                'complete_outfit_collage':True, 'fit_on_body_verified':False,
                'media_sha256':None})
        index=root/'index.json'
        index.write_text(json.dumps({'schema_version':1,'dataset_id':'image-teaste-2026-11-10','images':rows}))
        archive=root/'archive.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for i,r in enumerate(rows):
                if missing and i==2:continue
                content=b'\xff\xd8\xff'+(str(1 if duplicate and i==2 else i).encode()*6000)
                z.writestr('image-teaste/'+r['filename'],content)
            if bad_member:z.writestr('../leak.jpg',b'not actual data')
        return archive,index
    def test_import_46_unique_files_from_private_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,index=self.fixture(root)
            result=module.import_archive(archive,index,root/'out')
            self.assertEqual(result['media_count'],46)
            self.assertFalse(result['origin_verified'])
            self.assertTrue(all(not x['training_approved'] for x in result['images']))
            self.assertEqual(len(list((root/'out'/'images').glob('*.jpg'))),46)
    def test_incomplete_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,index=self.fixture(root,missing=True)
            with self.assertRaises(ValueError):module.import_archive(archive,index,root/'out')
    def test_duplicate_photos_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,index=self.fixture(root,duplicate=True)
            with self.assertRaises(ValueError):module.import_archive(archive,index,root/'out')
    def test_bad_archive_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive,index=self.fixture(root,bad_member=True)
            with self.assertRaises(ValueError):module.import_archive(archive,index,root/'out')

if __name__=='__main__':unittest.main()
