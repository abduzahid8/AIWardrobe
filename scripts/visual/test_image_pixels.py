import unittest
from PIL import Image,ImageDraw
from image_pixels import primary_hue
class HueEvidence(unittest.TestCase):
 def test_known_dominant_foreground_and_blank_abstention(self):
  image=Image.new('RGB',(200,200),'white');ImageDraw.Draw(image).rectangle((55,25,145,175),fill=(70,180,35))
  result=primary_hue(image)
  self.assertEqual(result['value'],'green');self.assertEqual(result['evidence'],'image-pixels');self.assertGreaterEqual(result['confidence'],0.8)
  self.assertIsNone(primary_hue(Image.new('RGB',(200,200),'white')))
 def test_mixed_hues_do_not_become_a_confident_primary(self):
  image=Image.new('RGB',(200,200),'white');d=ImageDraw.Draw(image);d.rectangle((40,25,99,175),fill='red');d.rectangle((100,25,159,175),fill='blue')
  self.assertIsNone(primary_hue(image))
if __name__=='__main__':unittest.main()
