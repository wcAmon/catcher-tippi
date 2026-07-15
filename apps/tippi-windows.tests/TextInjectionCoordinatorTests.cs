using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class TextInjectionCoordinatorTests
{
    [Fact]
    public void SubmitCommandIsHeldBackAndSendsOneReturn()
    {
        var injector = new FakeInjector();
        var coordinator = new TextInjectionCoordinator(injector);

        coordinator.Update("這是一段測試內容幫我送出");
        coordinator.Update("這是一段測試內容幫我送出");

        Assert.Equal("這是一段測試內容", injector.Text);
        Assert.Equal(1, injector.ReturnCount);
    }

    [Fact]
    public void FinishFlushesHeldContentWhenThereIsNoCommand()
    {
        var injector = new FakeInjector();
        var coordinator = new TextInjectionCoordinator(injector);

        coordinator.Update("普通的語音內容");
        coordinator.Finish();

        Assert.Equal("普通的語音內容", injector.Text);
        Assert.Equal(0, injector.ReturnCount);
    }

    private sealed class FakeInjector : ITextInjector
    {
        public string Text { get; private set; } = string.Empty;
        public int ReturnCount { get; private set; }

        public bool InsertText(string text)
        {
            Text += text;
            return true;
        }

        public bool PressEnter()
        {
            ReturnCount++;
            return true;
        }
    }
}
