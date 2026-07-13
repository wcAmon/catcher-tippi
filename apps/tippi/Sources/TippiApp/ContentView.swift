import AppKit
import SwiftUI
import TippiCore

struct ContentView: View {
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

    private var transcript: some View {
        let transcriptText = TranscriptFormatter.transcript(
            messages: controller.messages,
            names: controller.speakerNames
        )
        return VStack(alignment: .leading, spacing: 12) {
            Text("TRANSCRIPT")
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)
            ScrollView {
                Text(transcriptText.isEmpty ? placeholder : transcriptText)
                    .font(.system(size: 23, weight: .regular, design: .rounded))
                    .foregroundStyle(transcriptText.isEmpty ? .tertiary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.vertical, 4)
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
}
