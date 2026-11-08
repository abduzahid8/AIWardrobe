"""Conservative foreground hue evidence for plain-background catalog photos.
Abstains on non-uniform backgrounds, weak masks and mixed hues. No fiber/fit claims.
"""
import colorsys
import numpy as np
from PIL import Image

def primary_hue(image):
 a=np.asarray(image.resize((160,160)).convert('RGB'),dtype=np.float32)/255
 border=np.concatenate([a[:8].reshape(-1,3),a[-8:].reshape(-1,3),a[:, :8].reshape(-1,3),a[:, -8:].reshape(-1,3)])
 background=np.median(border,axis=0)
 if np.quantile(np.linalg.norm(border-background,axis=1),0.9)>0.15:return None
 mask=np.linalg.norm(a-background,axis=2)>0.16
 fraction=float(mask.mean())
 if not 0.08<=fraction<=0.78:return None
 foreground=a[mask];high=foreground.max(axis=1);low=foreground.min(axis=1)
 saturation=(high-low)/np.maximum(high,0.001)
 chromatic=foreground[(saturation>0.35)&(high>0.18)]
 if len(chromatic)<len(foreground)*0.6:return None
 labels=[]
 for rgb in chromatic:
  h,s,v=colorsys.rgb_to_hsv(*rgb);h*=360
  if 12<=h<65 and v<0.55:color='brown'
  elif h<12 or h>=345:color='red'
  elif h<42:color='orange'
  elif h<70:color='yellow'
  elif h<170:color='green'
  elif h<260:color='blue'
  elif h<305:color='purple'
  else:color='pink'
  labels.append(color)
 from collections import Counter
 ranked=Counter(labels).most_common();dominance=ranked[0][1]/len(labels);second=ranked[1][1]/len(labels) if len(ranked)>1 else 0
 if dominance<0.75 or dominance-second<0.4:return None
 return {'value':ranked[0][0],'confidence':round(0.8+0.15*dominance,5),'margin':round(dominance-second,5),'evidence':'image-pixels'}


def merge_primary_hue(attributes, observed):
 if not observed:return
 existing=attributes.get('color',{})
 if existing.get('evidence')=='human':return
 families={'navy':'blue','olive':'green','burgundy':'red','beige':'orange'}
 # A measured broad hue does not disprove an already supported finer shade.
 if existing.get('confidence',0)>=0.8 and families.get(existing.get('value'))==observed['value']:
  attributes['measuredHue']=observed
 else:attributes['color']=observed
