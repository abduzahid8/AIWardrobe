"""Dependency-free test of the ACTUAL SigLIP download function's transport.

Do not import torch/transformers or access the network. Compile the exact
production function with tiny fake HTTP/PIL implementations to enforce truthful
client identity, no auth/cookies and fail-closed source-photo behavior.
"""
import ast
import io
import pathlib
import time
import unittest

SOURCE=pathlib.Path(__file__).with_name('enrich_catalog.py')

class Reply:
    is_redirect=False
    headers={'content-type':'image/jpeg'}
    status_code=200
    ok=True
    def __init__(self,status=200):
        self.status_code=status
        self.ok=status==200
        self.closed=False
    def iter_content(self,size):
        yield b'\xff\xd8'+b'J'*2046
    def close(self):
        self.closed=True

class Images:
    width=300
    height=400
    def load(self):
        pass
    def convert(self,mode):
        assert mode=='RGB'
        return self

class ImageLib:
    @staticmethod
    def open(content):
        assert content.read(2)==b'\xff\xd8'
        return Images()

class ImageOps:
    @staticmethod
    def exif_transpose(image):
        return image

class Requests:
    def __init__(self,status=200):
        self.status=status
        self.calls=[]
        self.response=None
    def get(self,url,**kw):
        self.calls.append((url,kw))
        self.response=Reply(self.status)
        return self.response

def load_actual_fetch(status):
    tree=ast.parse(SOURCE.read_text())
    headers=next(node.value for node in tree.body
        if isinstance(node,ast.Assign) and any(
          isinstance(t,ast.Name) and t.id=='IMAGE_REQUEST_HEADERS'
          for t in node.targets))
    headers=ast.literal_eval(headers)
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and
      n.name=='fetch_image')
    requests=Requests(status)
    env=dict(IMAGE_REQUEST_HEADERS=headers,requests=requests,Image=ImageLib,
      ImageOps=ImageOps,io=io,time=time,public_url=lambda url:url)
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(SOURCE),'exec'),env)
    return env['fetch_image'],requests,headers

class TransportTest(unittest.TestCase):
    def test_identifying_original_photo_get_has_no_credentials_or_browser_impersonation(self):
        fetch,requests,headers=load_actual_fetch(200)
        data,image=fetch('https://shop.example/photo.jpg')
        self.assertEqual(len(data),2048)
        self.assertEqual((image.width,image.height),(300,400))
        self.assertEqual(headers,{'User-Agent':'AIWardrobe-Stylist6-Cloud/1.0',
            'Accept':'image/*,*/*;q=0.8'})
        self.assertEqual(requests.calls,[('https://shop.example/photo.jpg',{
            'headers':headers,'timeout':(5,10),
            'allow_redirects':False,'stream':True})])
        self.assertNotIn('Cookie',headers)
        self.assertNotIn('Authorization',headers)
        self.assertNotIn('Referer',headers)
        self.assertTrue(requests.response.closed)

    def test_http_403_stays_unavailable_and_no_other_credential_or_client_is_tried(self):
        fetch,requests,_=load_actual_fetch(403)
        with self.assertRaisesRegex(ValueError,'image HTTP 403'):
            fetch('https://shop.example/protected.jpg')
        self.assertEqual(len(requests.calls),1)
        self.assertTrue(requests.response.closed)

if __name__=='__main__':
    unittest.main()
