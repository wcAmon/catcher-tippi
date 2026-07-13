import SwiftUI
import TippiCore

struct MessageRow: View {
    let message: Message
    let name: String
    let accent: Color
    let lineText: String
    let onRename: (String) -> Void

    @State private var isRenaming = false
    @State private var draftName = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Button {
                    draftName = name
                    isRenaming = true
                } label: {
                    Text(name)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(accent)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $isRenaming, arrowEdge: .bottom) {
                    TextField("說話者名稱", text: $draftName)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180)
                        .padding(12)
                        .onSubmit {
                            onRename(draftName.trimmingCharacters(in: .whitespacesAndNewlines))
                            isRenaming = false
                        }
                }
                Text(TranscriptFormatter.timestamp(forMs: message.startMs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(message.text)
                .font(.system(size: 19, weight: .regular, design: .rounded))
                .foregroundStyle(message.isFinal ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contextMenu {
            Button("複製此則") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(lineText, forType: .string)
            }
        }
    }
}
