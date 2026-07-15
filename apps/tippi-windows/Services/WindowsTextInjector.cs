using System.ComponentModel;
using System.Runtime.InteropServices;

namespace Tippi.Windows.Services;

public sealed class WindowsTextInjector(nint applicationWindow) : ITextInjector
{
    private const uint InputKeyboard = 1;
    private const uint KeyEventKeyUp = 0x0002;
    private const uint KeyEventUnicode = 0x0004;
    private const ushort VirtualKeyReturn = 0x0D;

    public bool InsertText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return true;
        }
        if (!HasExternalTarget())
        {
            return false;
        }

        var inputs = new INPUT[text.Length * 2];
        int index = 0;
        foreach (char character in text)
        {
            inputs[index++] = KeyboardInput(0, character, KeyEventUnicode);
            inputs[index++] = KeyboardInput(0, character, KeyEventUnicode | KeyEventKeyUp);
        }
        return Send(inputs);
    }

    public bool PressEnter()
    {
        if (!HasExternalTarget())
        {
            return false;
        }
        return Send(
        [
            KeyboardInput(VirtualKeyReturn, '\0', 0),
            KeyboardInput(VirtualKeyReturn, '\0', KeyEventKeyUp),
        ]);
    }

    private bool HasExternalTarget()
    {
        nint foreground = GetForegroundWindow();
        return foreground != 0 && foreground != applicationWindow;
    }

    private static INPUT KeyboardInput(ushort virtualKey, char scanCode, uint flags)
    {
        return new INPUT
        {
            Type = InputKeyboard,
            Union = new InputUnion
            {
                Keyboard = new KEYBDINPUT
                {
                    VirtualKey = virtualKey,
                    ScanCode = scanCode,
                    Flags = flags,
                },
            },
        };
    }

    private static bool Send(INPUT[] inputs)
    {
        uint sent = SendInput((uint)inputs.Length, inputs, Marshal.SizeOf<INPUT>());
        if (sent == inputs.Length)
        {
            return true;
        }
        if (sent == 0 && Marshal.GetLastWin32Error() != 0)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Windows 無法將文字輸入到目前的程式。");
        }
        return false;
    }

    [DllImport("user32.dll")]
    private static extern nint GetForegroundWindow();

    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint SendInput(uint numberOfInputs, INPUT[] inputs, int sizeOfInput);

    [StructLayout(LayoutKind.Sequential)]
    private struct INPUT
    {
        public uint Type;
        public InputUnion Union;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct InputUnion
    {
        [FieldOffset(0)] public KEYBDINPUT Keyboard;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEYBDINPUT
    {
        public ushort VirtualKey;
        public ushort ScanCode;
        public uint Flags;
        public uint Time;
        public nuint ExtraInfo;
    }
}
