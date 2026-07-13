import AppKit
import SwiftUI
import TippiCore

struct VoiceInputTabView: View {
    @Environment(\.scenePhase) private var scenePhase
    @Bindable var controller: TranscriptionController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                permissionCard
                modelCard
                targetCard
                activityCard
                controls
            }
            .padding(36)
            .frame(maxWidth: .infinity, minHeight: 480, alignment: .topLeading)
        }
        .frame(minWidth: 600, minHeight: 480)
        .background(
            LinearGradient(
                colors: [Color(nsColor: .windowBackgroundColor), Color.indigo.opacity(0.06)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .task {
            await controller.prepareVoiceInput()
            controller.refreshAccessibility(prompt: false)
        }
        .onChange(of: scenePhase) {
            if scenePhase == .active {
                controller.refreshAccessibility(prompt: false)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 5) {
                Text("語音輸入")
                    .font(.system(size: 32, weight: .semibold, design: .rounded))
                Text("說話即可輸入文字，說出 Tippi Go 就會送出。")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("口令：Tippi Go")
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.indigo.opacity(0.12), in: Capsule())
        }
    }

    private var permissionCard: some View {
        statusCard(title: "輔助使用權限") {
            HStack(alignment: .center, spacing: 12) {
                Label(permissionText, systemImage: permissionSymbol)
                    .foregroundStyle(controller.accessibilityTrusted ? .green : .orange)
                Spacer()
                if !controller.accessibilityTrusted {
                    Button("要求輔助使用權限") {
                        controller.refreshAccessibility(prompt: true)
                    }
                    Button("開啟系統設定") {
                        openAccessibilitySettings()
                    }
                }
            }
        }
    }

    private var modelCard: some View {
        statusCard(title: "模型狀態") {
            VStack(alignment: .leading, spacing: 12) {
                transcriptionModelStatus
                Divider()
                keywordModelStatus
            }
        }
    }

    @ViewBuilder
    private var transcriptionModelStatus: some View {
        switch controller.state {
        case .modelMissing:
            Label("正在準備語音辨識模型", systemImage: "clock")
                .foregroundStyle(.secondary)
        case let .downloading(progress):
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: progress)
                Text("正在準備語音辨識模型 · \(progress.formatted(.percent.precision(.fractionLength(0))))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .loading:
            Label("正在載入語音辨識模型", systemImage: "gearshape.2")
                .foregroundStyle(.secondary)
        case .ready, .recording, .finishing:
            Label("語音辨識模型已就緒", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case let .failed(message):
            VStack(alignment: .leading, spacing: 10) {
                Label("語音辨識模型準備失敗", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                Button("重試語音辨識模型") {
                    Task { await controller.prepare() }
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    @ViewBuilder
    private var keywordModelStatus: some View {
        switch controller.voiceInputPreparation {
        case .notPrepared:
            Label("正在準備 Tippi Go 口令模型", systemImage: "clock")
                .foregroundStyle(.secondary)
        case let .downloading(progress):
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: progress)
                Text("正在下載並驗證口令模型 · \(progress.formatted(.percent.precision(.fractionLength(0))))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .loading:
            Label("正在載入口令模型", systemImage: "gearshape.2")
                .foregroundStyle(.secondary)
        case .ready:
            Label("Tippi Go 口令模型已就緒", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case let .failed(message):
            VStack(alignment: .leading, spacing: 10) {
                Label("口令模型準備失敗", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                Button("重試口令模型") {
                    Task { await controller.prepareVoiceInput() }
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var targetCard: some View {
        statusCard(title: "輸入目標") {
            VStack(alignment: .leading, spacing: 8) {
                Label(targetText, systemImage: "macwindow.on.rectangle")
                    .foregroundStyle(targetNeedsAttention ? .orange : .primary)
                if controller.isRecording(.voiceInput) && targetNeedsAttention {
                    Text("請切到目標輸入框；Tippi 不會把文字輸入到自己。")
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else {
                    Text("開始後切到要輸入文字的 App，並將游標放在輸入框。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var activityCard: some View {
        statusCard(title: "最近活動") {
            VStack(alignment: .leading, spacing: 8) {
                Label(controller.voiceInputMessage, systemImage: activitySymbol)
                    .foregroundStyle(activityColor)
                Text(
                    controller.lastInjectedText.isEmpty
                        ? "尚未注入文字"
                        : controller.lastInjectedText
                )
                .font(.body.monospaced())
                .foregroundStyle(controller.lastInjectedText.isEmpty ? .secondary : .primary)
                .lineLimit(3)
                .textSelection(.enabled)
            }
        }
    }

    private var controls: some View {
        HStack {
            Text(recordingHint)
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                controller.refreshAccessibility(prompt: false)
                Task { await controller.toggleRecording(mode: .voiceInput) }
            } label: {
                Label(
                    controller.isRecording(.voiceInput) ? "停止" : "開始語音輸入",
                    systemImage: controller.isRecording(.voiceInput) ? "stop.fill" : "waveform.and.mic"
                )
                .font(.headline)
                .frame(minWidth: 150)
                .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(controller.isRecording(.voiceInput) ? .red : .indigo)
            .disabled(!controller.canToggle(.voiceInput))
        }
    }

    private func statusCard<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.caption.weight(.semibold))
                .tracking(0.8)
                .foregroundStyle(.secondary)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(.background.opacity(0.74), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.separator.opacity(0.4)))
    }

    private var permissionText: String {
        controller.accessibilityTrusted ? "已允許跨 App 輸入" : "需要輔助使用權限"
    }

    private var permissionSymbol: String {
        controller.accessibilityTrusted ? "checkmark.shield.fill" : "lock.trianglebadge.exclamationmark"
    }

    private var targetText: String {
        controller.targetApplicationName.map { "目前目標：\($0)" } ?? "目前沒有可用目標"
    }

    private var targetNeedsAttention: Bool {
        controller.targetApplicationName == nil || controller.targetApplicationName == "Tippi"
    }

    private var activitySymbol: String {
        if controller.voiceInputMessage == "已送出" { return "paperplane.fill" }
        return controller.isRecording(.voiceInput) ? "waveform" : "info.circle"
    }

    private var activityColor: Color {
        controller.voiceInputMessage == "已送出" ? .green : .secondary
    }

    private var recordingHint: String {
        if controller.isRecording(.voiceInput) { return "正在聆聽；說 Tippi Go 送出。" }
        if controller.activeMode == .transcription { return "轉錄分頁正在使用麥克風。" }
        if !controller.accessibilityTrusted { return "授權後才能開始。" }
        if case .failed = controller.state { return "請先重試語音辨識模型。" }
        if controller.state != .ready { return "語音辨識模型就緒後才能開始。" }
        if controller.voiceInputPreparation != .ready { return "口令模型就緒後才能開始。" }
        return "音訊與辨識都留在這台 Mac。"
    }

    private func openAccessibilitySettings() {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ) else { return }
        NSWorkspace.shared.open(url)
    }
}
