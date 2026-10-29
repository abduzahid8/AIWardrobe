from pathlib import Path
import json,hashlib,sys,time
root=Path('/Users/v.zahid/Library/Developer/CoreSimulator/Devices/BBE8D6A5-D0B5-4DE2-8808-EC5BBEB1B133/data/Containers/Data/Application/95E2D36E-0FF1-4A54-83D2-2921F165AF66/Documents/ExponentExperienceData/@abduzahid8/wardrobe-app/RCTAsyncLocalStorage')
manifest=json.load(open(root/'manifest.json'));matches=[k for k in manifest if k.startswith('stylistRotationDecks::') and k.endswith('|shop|old_money|date')]
if len(matches)!=1:raise SystemExit('Cannot identify exact Old Money / Date scope')
key=matches[0];raw=manifest[key]
if raw is None:raw=(root/hashlib.md5(key.encode()).hexdigest()).read_text()
decks=json.loads(raw);out=Path('/tmp/stylist-shoe-rotation');snapshot={'scope':key.split('::',1)[1],'capturedAt':time.time(),'decks':decks}
if '--before' in sys.argv:
 (out/'history-before.json').write_text(json.dumps(snapshot,indent=2));print('Saved existing history; no cache reset:',len(decks),'decks')
else:
 before=json.load(open(out/'history-before.json'));ids={d['requestId'] for d in before['decks']};new=[d for d in decks if d['requestId'] not in ids]
 (out/'history-after.json').write_text(json.dumps({**snapshot,'decks':new,'olderBeforeRun':before['decks']},indent=2));print('Accepted final displayed decks since boundary:',len(new));print('Request IDs:',[d['requestId'] for d in new])
