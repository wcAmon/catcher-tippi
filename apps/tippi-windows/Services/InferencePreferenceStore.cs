using System.IO;

namespace Tippi.Windows.Services;

public sealed class InferencePreferenceStore
{
    private readonly string _path;

    public InferencePreferenceStore(string? path = null)
    {
        _path = path ?? DefaultPath;
    }

    public InferenceBackendPreference Load()
    {
        try
        {
            string value = File.ReadAllText(_path).Trim();
            return Enum.TryParse(value, ignoreCase: true, out InferenceBackendPreference preference)
                ? preference
                : InferenceBackendPreference.Auto;
        }
        catch (IOException)
        {
            return InferenceBackendPreference.Auto;
        }
        catch (UnauthorizedAccessException)
        {
            return InferenceBackendPreference.Auto;
        }
    }

    public void Save(InferenceBackendPreference preference)
    {
        try
        {
            string? directory = Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllText(_path, preference.ToString());
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static string DefaultPath
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "backend-preference.txt");
        }
    }
}
