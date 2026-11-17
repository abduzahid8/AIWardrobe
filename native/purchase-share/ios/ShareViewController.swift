import UIKit
import Social
import UniformTypeIdentifiers

// Add this file to a separately signed Share Extension target. No containing-app
// launch hacks: save a bounded, owner-tagged inbox item and let the user open it.
final class ShareViewController: SLComposeServiceViewController {
  private let suite = "group.com.aiwardrobe.purchase"
  override func isContentValid() -> Bool { return true }
  override func didSelectPost() {
    let attachments = (extensionContext?.inputItems as? [NSExtensionItem] ?? []).flatMap { $0.attachments ?? [] }
    guard let provider = attachments.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.url.identifier) || $0.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) }) else {
      extensionContext?.completeRequest(returningItems: nil); return
    }
    guard let sharedDefaults = UserDefaults(suiteName: suite), let sourceOwner = sharedDefaults.string(forKey: "purchaseOwner") else {
      extensionContext?.completeRequest(returningItems: nil); return
    }
    let type = provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) ? UTType.url.identifier : UTType.plainText.identifier
    provider.loadItem(forTypeIdentifier: type, options: nil) { [weak self] value, _ in
      guard let self = self, let defaults = UserDefaults(suiteName: self.suite), let owner = defaults.string(forKey: "purchaseOwner"), owner == sourceOwner else {
        DispatchQueue.main.async { self?.extensionContext?.completeRequest(returningItems: nil) }; return
      }
      let text = (value as? URL)?.absoluteString ?? (value as? String) ?? ""
      if !text.isEmpty && text.utf8.count <= 4096 {
        defaults.set(["owner": owner, "input": text, "createdAt": Date().timeIntervalSince1970], forKey: "purchaseInbox")
      }
      DispatchQueue.main.async { self.extensionContext?.completeRequest(returningItems: nil) }
    }
  }
}
