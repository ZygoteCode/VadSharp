using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace VadSharp
{
    public class VadDetector
    {
        private readonly VadOnnxModel _model;
        private readonly float _threshold;
        private readonly float _negThreshold;
        private readonly int _samplingRate;
        private readonly int _windowSizeSample;
        private readonly float _minSpeechSamples;
        private readonly float _speechPadSamples;
        private readonly float _maxSpeechSamples;
        private readonly float _minSilenceSamples;
        private readonly float _minSilenceSamplesAtMaxSpeech;
        private int _audioLengthSamples;
        private const float THRESHOLD_GAP = 0.15f;
        private const int SAMPLING_RATE_8K = 8000;
        private const int SAMPLING_RATE_16K = 16000;

        public VadDetector(string onnxModelPath, float threshold, int samplingRate, int minSpeechDurationMs, float maxSpeechDurationSeconds, int minSilenceDurationMs, int speechPadMs)
        {
            if (samplingRate != SAMPLING_RATE_8K && samplingRate != SAMPLING_RATE_16K)
            {
                throw new ArgumentException("Sampling rate not supported, only available for [8000, 16000]");
            }

            _model = new VadOnnxModel(onnxModelPath);
            _samplingRate = samplingRate;
            _threshold = threshold;
            _negThreshold = threshold - THRESHOLD_GAP;
            _windowSizeSample = samplingRate == SAMPLING_RATE_16K ? 512 : 256;
            _minSpeechSamples = samplingRate * minSpeechDurationMs / 1000f;
            _speechPadSamples = samplingRate * speechPadMs / 1000f;
            _maxSpeechSamples = samplingRate * maxSpeechDurationSeconds - _windowSizeSample - 2 * _speechPadSamples;
            _minSilenceSamples = samplingRate * minSilenceDurationMs / 1000f;
            _minSilenceSamplesAtMaxSpeech = samplingRate * 98 / 1000f;
            Reset();
        }

        public void Reset() => _model.ResetStates();

        public List<VadSpeechSegment> GetSpeechSegmentList(string wavFile) => GetSpeechSegmentList(new FileInfo(wavFile));

        public List<VadSpeechSegment> GetSpeechSegmentList(FileInfo wavFile)
        {
            Reset();
            using var reader = new AudioFileReader(wavFile.FullName);
            var resampler = new WdlResamplingSampleProvider(reader, _samplingRate);
            var speechProbList = new List<float>();
            _audioLengthSamples = (int)(reader.Length / 2);
            int window = _windowSizeSample;
            float[] buffer = new float[window];

            while (resampler.Read(buffer, 0, window) > 0)
            {
                float speechProb = _model.Call(new[] { buffer }, _samplingRate)[0];
                speechProbList.Add(speechProb);
            }

            return CalculateProb(speechProbList);
        }

        private List<VadSpeechSegment> CalculateProb(List<float> speechProbList)
        {
            var result = new List<VadSpeechSegment>();
            bool triggered = false;
            int tempEnd = 0, prevEnd = 0, nextStart = 0;
            var segment = new VadSpeechSegment();
            int window = _windowSizeSample;

            for (int i = 0, count = speechProbList.Count; i < count; i++)
            {
                float prob = speechProbList[i];
                int currentOffset = window * i;

                if (prob >= _threshold && tempEnd != 0)
                {
                    tempEnd = 0;

                    if (nextStart < prevEnd)
                    {
                        nextStart = currentOffset;
                    }
                }

                if (prob >= _threshold && !triggered)
                {
                    triggered = true;
                    segment.StartOffset = currentOffset;
                    continue;
                }

                if (triggered && currentOffset - segment.StartOffset > _maxSpeechSamples)
                {
                    if (prevEnd != 0)
                    {
                        segment.EndOffset = prevEnd;
                        result.Add(segment);
                        segment = new VadSpeechSegment();

                        if (nextStart < prevEnd)
                        {
                            triggered = false;
                        }
                        else
                        {
                            segment.StartOffset = nextStart;
                        }

                        prevEnd = nextStart = tempEnd = 0;
                    }
                    else
                    {
                        segment.EndOffset = currentOffset;
                        result.Add(segment);
                        segment = new VadSpeechSegment();
                        prevEnd = nextStart = tempEnd = 0;
                        triggered = false;
                        continue;
                    }
                }

                if (prob < _negThreshold && triggered)
                {
                    if (tempEnd == 0)
                    {
                        tempEnd = currentOffset;
                    }

                    if (currentOffset - tempEnd > _minSilenceSamplesAtMaxSpeech)
                    {
                        prevEnd = tempEnd;
                    }

                    if (currentOffset - tempEnd < _minSilenceSamples)
                    {
                        continue;
                    }
                    else
                    {
                        segment.EndOffset = tempEnd;

                        if ((segment.EndOffset - segment.StartOffset) > _minSpeechSamples)
                        {
                            result.Add(segment);
                        }

                        segment = new VadSpeechSegment();
                        prevEnd = nextStart = tempEnd = 0;
                        triggered = false;
                        continue;
                    }
                }
            }

            if (segment.StartOffset != null && (_audioLengthSamples - segment.StartOffset) > _minSpeechSamples)
            {
                segment.EndOffset = _audioLengthSamples;
                result.Add(segment);
            }

            for (int i = 0, count = result.Count; i < count; i++)
            {
                var item = result[i];

                if (i == 0)
                {
                    item.StartOffset = Math.Max(0, item.StartOffset.Value - (int)_speechPadSamples);
                }

                if (i != count - 1)
                {
                    var nextItem = result[i + 1];
                    int silenceDuration = nextItem.StartOffset.Value - item.EndOffset.Value;

                    if (silenceDuration < 2 * _speechPadSamples)
                    {
                        int halfSilence = silenceDuration / 2;
                        item.EndOffset += halfSilence;
                        nextItem.StartOffset = Math.Max(0, nextItem.StartOffset.Value - halfSilence);
                    }
                    else
                    {
                        item.EndOffset = Math.Min(_audioLengthSamples, item.EndOffset.Value + (int)_speechPadSamples);
                        nextItem.StartOffset = Math.Max(0, nextItem.StartOffset.Value - (int)_speechPadSamples);
                    }
                }
                else
                {
                    item.EndOffset = Math.Min(_audioLengthSamples, item.EndOffset.Value + (int)_speechPadSamples);
                }
            }

            return MergeListAndCalculateSecond(result, _samplingRate);
        }

        private List<VadSpeechSegment> MergeListAndCalculateSecond(List<VadSpeechSegment> segments, int samplingRate)
        {
            var merged = new List<VadSpeechSegment>();

            if (segments == null || segments.Count == 0)
            {
                return merged;
            }

            segments.Sort((a, b) => a.StartOffset.Value.CompareTo(b.StartOffset.Value));
            int left = segments[0].StartOffset.Value;
            int right = segments[0].EndOffset.Value;

            for (int i = 1, count = segments.Count; i < count; i++)
            {
                var seg = segments[i];

                if (seg.StartOffset > right)
                {
                    merged.Add(new VadSpeechSegment(left, right, CalculateSecondByOffset(left, samplingRate), CalculateSecondByOffset(right, samplingRate)));
                    left = seg.StartOffset.Value;
                    right = seg.EndOffset.Value;
                }
                else
                {
                    right = Math.Max(right, seg.EndOffset.Value);
                }
            }

            merged.Add(new VadSpeechSegment(left, right, CalculateSecondByOffset(left, samplingRate), CalculateSecondByOffset(right, samplingRate)));
            return merged;
        }

        private float CalculateSecondByOffset(int offset, int samplingRate)
        {
            float seconds = offset / (float)samplingRate;
            return (float)(Math.Floor(seconds * 1000.0f) / 1000.0f);
        }
    }
}