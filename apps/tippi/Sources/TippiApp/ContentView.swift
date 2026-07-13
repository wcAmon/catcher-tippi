import SwiftUI
import TippiCore

struct ContentView: View {
    @Bindable var controller: TranscriptionController

    var body: some View {
        TabView {
            TranscriptionTabView(controller: controller)
                .tabItem { Label("轉錄", systemImage: "text.bubble") }
            VoiceInputPlaceholderView()
                .tabItem { Label("語音輸入", systemImage: "keyboard") }
        }
        .frame(minWidth: 600, minHeight: 480)
    }
}
