import Foundation

/// German for German systems, English otherwise. The interface itself (web) has its own language switch.
func L(_ german: String, _ english: String) -> String {
    (Locale.preferredLanguages.first ?? "en").hasPrefix("de") ? german : english
}
