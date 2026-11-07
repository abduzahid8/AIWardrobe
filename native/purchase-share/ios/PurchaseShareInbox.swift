import Foundation
import React

@objc(PurchaseShareInbox)
final class PurchaseShareInbox: NSObject {
  private let defaults = UserDefaults(suiteName: "group.com.aiwardrobe.purchase")
  @objc static func requiresMainQueueSetup() -> Bool { return false }
  @objc func setOwner(_ owner: String) {
    if owner.isEmpty { defaults?.removeObject(forKey: "purchaseOwner"); defaults?.removeObject(forKey: "purchaseInbox") }
    else { defaults?.set(owner, forKey: "purchaseOwner") }
  }
  @objc func consume(_ owner: String, resolver resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    guard let data = defaults?.dictionary(forKey: "purchaseInbox") else { resolve(nil); return }
    defaults?.removeObject(forKey: "purchaseInbox")
    guard data["owner"] as? String == owner, let text = data["input"] as? String,
      text.utf8.count <= 4096, let timestamp = data["createdAt"] as? Double,
      Date().timeIntervalSince1970 - timestamp >= 0, Date().timeIntervalSince1970 - timestamp < 86400 else { resolve(nil); return }
    resolve(text)
  }
}
