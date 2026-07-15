using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Tippi.Windows.Services;

internal static class GeneratorRuntimeOptions
{
    private static readonly FieldInfo GeneratorHandleField = typeof(Generator).GetField(
        "_generatorHandle",
        BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingFieldException(typeof(Generator).FullName, "_generatorHandle");

    public static void Set(Generator generator, string key, string value)
    {
        nint handle = (nint)(GeneratorHandleField.GetValue(generator)
            ?? throw new InvalidOperationException("ONNX Runtime generator handle is unavailable."));
        nint result = OgaGeneratorSetRuntimeOption(handle, Utf8(key), Utf8(value));
        if (result == 0)
        {
            return;
        }

        try
        {
            string message = Marshal.PtrToStringUTF8(OgaResultGetError(result))
                ?? "ONNX Runtime rejected the language option.";
            throw new InvalidOperationException(message);
        }
        finally
        {
            OgaDestroyResult(result);
        }
    }

    private static byte[] Utf8(string value) => Encoding.UTF8.GetBytes(value + '\0');

    [DllImport("onnxruntime-genai", EntryPoint = "OgaGenerator_SetRuntimeOption", CallingConvention = CallingConvention.Winapi)]
    private static extern nint OgaGeneratorSetRuntimeOption(nint generator, byte[] key, byte[] value);

    [DllImport("onnxruntime-genai", CallingConvention = CallingConvention.Winapi)]
    private static extern nint OgaResultGetError(nint result);

    [DllImport("onnxruntime-genai", CallingConvention = CallingConvention.Winapi)]
    private static extern void OgaDestroyResult(nint result);
}
