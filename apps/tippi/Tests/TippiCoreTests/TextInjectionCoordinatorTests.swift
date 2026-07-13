import Testing
@testable import TippiCore

@MainActor
private final class FakeTextInjector: TextInjecting {
    enum Event: Equatable {
        case text(String)
        case returnKey
    }

    var trusted = true
    var events: [Event] = []

    func isTrusted(prompt: Bool) -> Bool {
        trusted
    }

    func inject(_ text: String) throws {
        events.append(.text(text))
    }

    func submit() throws {
        events.append(.returnKey)
    }
}

@MainActor
private final class FakeFrontmostApplicationProvider: FrontmostApplicationProviding {
    var application: TargetApplication?

    init(application: TargetApplication?) {
        self.application = application
    }

    func current() -> TargetApplication? {
        application
    }
}

private let tippiBundleIdentifier = "com.wcamon.tippi"
private let targetApplication = TargetApplication(
    name: "TextEdit",
    bundleIdentifier: "com.apple.TextEdit"
)

@MainActor
private func makeCoordinator(
    target: TargetApplication? = targetApplication
) -> (TextInjectionCoordinator, FakeTextInjector, FakeFrontmostApplicationProvider) {
    let injector = FakeTextInjector()
    let targetProvider = FakeFrontmostApplicationProvider(application: target)
    let coordinator = TextInjectionCoordinator(
        injector: injector,
        targetProvider: targetProvider,
        ownBundleIdentifier: tippiBundleIdentifier
    )
    return (coordinator, injector, targetProvider)
}

@MainActor
@Test
func appendOnlyUpdatesInjectOnlyNewSuffix() throws {
    let (coordinator, injector, _) = makeCoordinator()

    #expect(try coordinator.consume("你好") == .injected(text: "你好", target: "TextEdit"))
    #expect(try coordinator.consume("你好，世界") == .injected(text: "，世界", target: "TextEdit"))
    #expect(try coordinator.consume("你好，世界") == .noChange)
    #expect(injector.events == [.text("你好"), .text("，世界")])
}

@MainActor
@Test
func divergentPrefixThrowsWithoutBackspaceOrSubmit() throws {
    let (coordinator, injector, _) = makeCoordinator()
    _ = try coordinator.consume("alpha")

    #expect(throws: TextInjectionError.divergentPrefix(
        previous: "alpha",
        current: "alpine"
    )) {
        try coordinator.consume("alpine")
    }
    #expect(injector.events == [.text("alpha")])
}

@MainActor
@Test
func submitInjectsRemainingSuffixBeforeReturn() throws {
    let (coordinator, injector, _) = makeCoordinator()
    _ = try coordinator.consume("你好")

    #expect(
        try coordinator.submit("你好，世界")
            == .submitted(text: "你好，世界", target: "TextEdit")
    )
    #expect(injector.events == [
        .text("你好"),
        .text("，世界"),
        .returnKey,
    ])
}

@MainActor
@Test
func duplicateSubmitIsIgnoredUntilReset() throws {
    let (coordinator, injector, _) = makeCoordinator()

    _ = try coordinator.submit("第一輪")
    #expect(try coordinator.submit("第一輪") == .duplicateCommandIgnored)
    #expect(injector.events == [.text("第一輪"), .returnKey])

    coordinator.resetTurn()
    _ = try coordinator.submit("第二輪")
    #expect(injector.events == [
        .text("第一輪"),
        .returnKey,
        .text("第二輪"),
        .returnKey,
    ])
}

@MainActor
@Test
func tippiFrontmostDoesNotInjectOrSubmit() throws {
    let (coordinator, injector, targetProvider) = makeCoordinator(
        target: TargetApplication(name: "Tippi", bundleIdentifier: tippiBundleIdentifier)
    )

    #expect(try coordinator.consume("不要打回自己") == .waitingForTarget)
    #expect(try coordinator.submit("不要送出") == .waitingForTarget)
    #expect(injector.events.isEmpty)

    targetProvider.application = targetApplication
    #expect(
        try coordinator.consume("切換後才注入")
            == .injected(text: "切換後才注入", target: "TextEdit")
    )
}

@MainActor
@Test
func unicodeTextIsPassedWithoutClipboardTransformation() throws {
    let (coordinator, injector, _) = makeCoordinator()
    let text = "繁體中文 e\u{301} 👩🏽‍💻\n第二行"

    _ = try coordinator.submit(text)

    #expect(injector.events == [.text(text), .returnKey])
}
