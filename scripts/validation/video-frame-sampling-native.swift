import Foundation
import AVFoundation
import CoreGraphics

@main struct VideoSampleProbe {
  static func main() async throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
    let duration = try await asset.load(.duration).seconds
    var samples: [[String: Any]] = []
    if CommandLine.arguments.count > 2 {
      let generator = AVAssetImageGenerator(asset: asset)
      generator.appliesPreferredTrackTransform = true
      generator.requestedTimeToleranceAfter = .zero
      generator.requestedTimeToleranceBefore = .zero
      for value in CommandLine.arguments[2].split(separator: ",") {
        let ms = Int64(value)!
        let image = try generator.copyCGImage(at: CMTime(value: ms, timescale: 1000), actualTime: nil)
        var pixel: [UInt8] = [0,0,0,0]
        let rgb = CGColorSpaceCreateDeviceRGB()
        pixel.withUnsafeMutableBytes { bytes in
          let context = CGContext(data: bytes.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
            bytesPerRow: 4, space: rgb, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
          context.draw(image, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        samples.append(["timeMs": ms, "rgb": Array(pixel.prefix(3))])
      }
    }
    let result: [String: Any] = ["durationSeconds": duration, "samples": samples]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
  }
}
