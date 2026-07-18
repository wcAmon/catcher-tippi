using System.Text.RegularExpressions;

namespace Tippi.Windows.Services;

public interface ITextInjector
{
    bool InsertText(string text);
    bool PressEnter();
}

public sealed partial class TextInjectionCoordinator
{
    public const string SubmitCommand = "幫我送出";
    private const int SafetyTailCharacters = 2;
    private readonly ITextInjector _injector;
    private int _injectedLength;
    private bool _returnSent;
    private bool _submitRequested;
    private int _contentLengthWhenSubmitRequested;
    private string _latestText = string.Empty;

    public TextInjectionCoordinator(ITextInjector injector)
    {
        _injector = injector;
    }

    public void Reset()
    {
        _injectedLength = 0;
        _returnSent = false;
        _submitRequested = false;
        _contentLengthWhenSubmitRequested = 0;
        _latestText = string.Empty;
    }

    public void Update(string fullText)
    {
        if (_returnSent || fullText.Length < _injectedLength)
        {
            return;
        }
        _latestText = fullText;

        Match command = SubmitCommandAtEnd().Match(fullText);
        if (command.Success)
        {
            if (_submitRequested)
            {
                SubmitThrough(command.Index);
            }
            return;
        }

        int safeLength = Math.Max(0, fullText.Length - SubmitCommand.Length - SafetyTailCharacters);
        InjectThrough(fullText, safeLength);
    }

    public void RequestSubmit()
    {
        if (_returnSent)
        {
            return;
        }
        if (!_submitRequested)
        {
            _contentLengthWhenSubmitRequested = _latestText.Length;
            _submitRequested = true;
        }
        Match command = SubmitCommandAtEnd().Match(_latestText);
        if (command.Success)
        {
            SubmitThrough(command.Index);
        }
    }

    public void Finish()
    {
        if (_returnSent)
        {
            return;
        }
        if (_submitRequested)
        {
            Match command = SubmitCommandAtEnd().Match(_latestText);
            int contentLength = command.Success
                ? command.Index
                : Math.Max(_injectedLength, _contentLengthWhenSubmitRequested);
            SubmitThrough(contentLength);
        }
        else
        {
            InjectThrough(_latestText, _latestText.Length);
        }
    }

    private void SubmitThrough(int contentLength)
    {
        if (contentLength <= 0 && _injectedLength <= 0)
        {
            return;
        }
        if (InjectThrough(_latestText, contentLength) && _injector.PressEnter())
        {
            _returnSent = true;
        }
    }

    private bool InjectThrough(string text, int endExclusive)
    {
        if (endExclusive <= _injectedLength)
        {
            return true;
        }
        string delta = text[_injectedLength..endExclusive];
        if (!_injector.InsertText(delta))
        {
            return false;
        }
        _injectedLength = endExclusive;
        return true;
    }

    [GeneratedRegex("幫我送出[\\s，。！？、,.!?]*$", RegexOptions.CultureInvariant)]
    private static partial Regex SubmitCommandAtEnd();
}
