import AppKit
import SwiftUI
import TippiCore
import UniformTypeIdentifiers

struct TranscriptionTabView: View {
    @Bindable var controller: TranscriptionController

    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            header
            transcript
            footer
        }
        .padding(36)
        .frame(minWidth: 600, minHeight: 480)
        .background(
            LinearGradient(
                colors: [Color(nsColor: .windowBackgroundColor), Color.blue.opacity(0.055)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("tippi")
                    .font(.system(size: 34, weight: .semibold, design: .rounded))
                Text("Private speech to text, powered locally by Catcher")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Label(statusText, systemImage: statusSymbol)
                .font(.callout.weight(.medium))
                .foregroundStyle(statusColor)
                .accessibilityLabel("Status: \(statusText)")
        }
    }

    private static let accents: [Color] = [.blue, .green, .orange, .purple]

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TRANSCRIPT")
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)
            if controller.warningMessage != nil {
                Label("說話者分離已暫停,文字繼續轉寫", systemImage: "person.2.slash")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            }
            if controller.messages.isEmpty {
                Text(placeholder)
                    .font(.system(size: 23, weight: .regular, design: .rounded))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(controller.messages) { message in
                                MessageRow(
                                    message: message,
                                    name: TranscriptFormatter.displayName(
                                        for: message.speaker,
                                        names: controller.speakerNames
                                    ),
                                    accent: Self.accents[message.speaker % Self.accents.count],
                                    lineText: TranscriptFormatter.line(for: message, names: controller.speakerNames),
                                    onRename: { newName in
                                        rename(speaker: message.speaker, to: newName)
                                    }
                                )
                                .id(message.id)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .onChange(of: controller.messages.last?.text) {
                        if let lastID = controller.messages.last?.id {
                            proxy.scrollTo(lastID, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.background.opacity(0.72), in: RoundedRectangle(cornerRadius: 20))
        .overlay(RoundedRectangle(cornerRadius: 20).stroke(.separator.opacity(0.45)))
    }

    @ViewBuilder
    private var footer: some View {
        switch controller.state {
        case let .downloading(progress):
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: progress)
                Text("Downloading Catcher model · \(progress.formatted(.percent.precision(.fractionLength(0))))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case let .failed(message):
            HStack {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                Spacer()
                Button("Microphone Settings") { openMicrophoneSettings() }
                Button("Retry") { Task { await controller.prepare() } }
                    .buttonStyle(.borderedProminent)
            }
        default:
            HStack {
                Text(controller.isRecording ? "Listening and transcribing…" : "Your audio stays on this Mac")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("複製全部") { copyAll() }
                    .disabled(controller.messages.isEmpty)
                Button("匯出…") { exportTranscript() }
                    .disabled(controller.messages.isEmpty)
                Button {
                    Task { await controller.toggleRecording() }
                } label: {
                    Label(
                        controller.isRecording ? "Stop Recording" : "Start Recording",
                        systemImage: controller.isRecording ? "stop.fill" : "mic.fill"
                    )
                    .font(.headline)
                    .frame(minWidth: 150)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(controller.isRecording ? .red : .blue)
                .disabled(controller.state != .ready && controller.state != .recording)
                .keyboardShortcut(.space, modifiers: [])
            }
        }
    }

    private var placeholder: String {
        switch controller.state {
        case .recording: "Listening…"
        default: "Turn recording on and start speaking."
        }
    }

    private func rename(speaker: Int, to newName: String) {
        if newName.isEmpty {
            controller.speakerNames.removeValue(forKey: speaker)
        } else {
            controller.speakerNames[speaker] = newName
        }
    }

    private var statusText: String {
        switch controller.state {
        case .modelMissing: "Model needed"
        case .downloading: "Downloading"
        case .loading: "Loading Catcher"
        case .ready: "Ready"
        case .recording: "Recording on"
        case .finishing: "Finishing"
        case .failed: "Needs attention"
        }
    }

    private var statusSymbol: String {
        switch controller.state {
        case .ready: "checkmark.circle.fill"
        case .recording: "waveform.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        default: "circle.dotted"
        }
    }

    private var statusColor: Color {
        switch controller.state {
        case .ready: .green
        case .recording: .red
        case .failed: .orange
        default: .secondary
        }
    }

    private func openMicrophoneSettings() {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        ) else { return }
        NSWorkspace.shared.open(url)
    }

    private var fullTranscript: String {
        TranscriptFormatter.transcript(
            messages: controller.messages,
            names: controller.speakerNames
        )
    }

    private func copyAll() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullTranscript, forType: .string)
    }

    private func exportTranscript() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText, .json]
        panel.nameFieldStringValue = "Tippi 逐字稿.txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try exportData(for: url).write(to: url)
        } catch {
            let alert = NSAlert()
            alert.alertStyle = .warning
            alert.messageText = "匯出失敗"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    private func exportData(for url: URL) throws -> Data {
        if url.pathExtension.lowercased() == "json" {
            return try TranscriptJSONExporter.data(
                messages: controller.messages,
                names: controller.speakerNames
            )
        }
        return Data(fullTranscript.utf8)
    }
}
