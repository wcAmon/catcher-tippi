using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class InferencePreferenceStoreTests
{
    [Fact]
    public void PreferenceRoundTripsAndInvalidContentReturnsAuto()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"tippi-preference-{Guid.NewGuid():N}");
        string path = Path.Combine(directory, "preference.txt");
        try
        {
            var store = new InferencePreferenceStore(path);
            Assert.Equal(InferenceBackendPreference.Auto, store.Load());

            store.Save(InferenceBackendPreference.DirectML);
            Assert.Equal(InferenceBackendPreference.DirectML, store.Load());

            File.WriteAllText(path, "not-a-backend");
            Assert.Equal(InferenceBackendPreference.Auto, store.Load());
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }
}
