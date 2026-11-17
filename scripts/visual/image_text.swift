// Offline image-quality corroboration. Emits geometry/counts, never OCR text.
import Foundation
import Vision
import ImageIO
let url=URL(fileURLWithPath:CommandLine.arguments[1])
let request=VNRecognizeTextRequest()
request.recognitionLevel = .fast
request.usesLanguageCorrection = false
try VNImageRequestHandler(url:url).perform([request])
let boxes=(request.results ?? []).filter { ($0.topCandidates(1).first?.confidence ?? 0)>0.7 }
let coverage=boxes.reduce(0.0) { $0 + Double($1.boundingBox.width*$1.boundingBox.height) }
let data=try JSONSerialization.data(withJSONObject:["textCoverage":min(1.0,coverage),"textRegions":boxes.count])
print(String(data:data,encoding:.utf8)!)
