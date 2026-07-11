import Testing
@testable import TippiCore

@Test
func acceptsTheModelAndRecordingHappyPath() throws {
    var machine = TippiStateMachine(initial: .modelMissing)

    try machine.transition(to: .downloading(0.2))
    try machine.transition(to: .downloading(0.8))
    try machine.transition(to: .loading)
    try machine.transition(to: .ready)
    try machine.transition(to: .recording)
    try machine.transition(to: .finishing)
    try machine.transition(to: .ready)

    #expect(machine.state == .ready)
}

@Test
func rejectsRecordingBeforeTheModelIsReady() {
    var machine = TippiStateMachine(initial: .modelMissing)

    #expect(throws: TippiStateError.self) {
        try machine.transition(to: .recording)
    }
    #expect(machine.state == .modelMissing)
}

@Test
func failureCanRetryFromTheLastKnownModelState() throws {
    var machine = TippiStateMachine(initial: .ready)
    try machine.transition(to: .failed("microphone denied"))
    try machine.transition(to: .ready)
    #expect(machine.state == .ready)
}
