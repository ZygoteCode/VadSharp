using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VadSharp
{
    public class VadOnnxModel : IDisposable
    {
        private readonly InferenceSession session;
        private float[][][] state;
        private float[][] context;
        private int lastSr, lastBatchSize;
        private static readonly int[] SupportedRates = { 8000, 16000 };

        public VadOnnxModel(string modelPath)
        {
            var sessionOptions = new SessionOptions();

            try
            {
                sessionOptions.AppendExecutionProvider_DML();
            }
            catch
            {

            }

            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = 1;
            sessionOptions.EnableCpuMemArena = true;

            session = new InferenceSession(modelPath, sessionOptions);
            ResetStates();
        }

        public void ResetStates()
        {
            state = new float[2][][]
            {
                new float[1][] { new float[128] },
                new float[1][] { new float[128] }
            };
            context = Array.Empty<float[]>();
            lastSr = 0;
            lastBatchSize = 0;
        }

        public void Dispose() => session?.Dispose();

        public class ValidationResult
        {
            public float[][] X { get; }
            public int Sr { get; }
            public ValidationResult(float[][] x, int sr)
            {
                X = x;
                Sr = sr;
            }
        }

        private ValidationResult ValidateInput(float[][] x, int sr)
        {
            if (x.Length == 1)
            {
                x = new float[][] { x[0] };
            }

            if (x.Length > 2)
            {
                throw new ArgumentException($"Incorrect audio data dimension: {x[0].Length}");
            }

            if (sr != 16000 && sr % 16000 == 0)
            {
                int step = sr / 16000;

                for (int i = 0; i < x.Length; i++)
                {
                    float[] current = x[i];
                    int newLength = (current.Length + step - 1) / step;
                    float[] newArr = new float[newLength];

                    for (int j = 0, index = 0; j < current.Length; j += step, index++)
                    {
                        newArr[index] = current[j];
                    }

                    x[i] = newArr;
                }

                sr = 16000;
            }

            if (Array.IndexOf(SupportedRates, sr) < 0)
            {
                throw new ArgumentException($"Only supports sample rates {string.Join(", ", SupportedRates)} (or multiples of 16000)");
            }

            if (((float)sr) / x[0].Length > 31.25)
            {
                throw new ArgumentException("Input audio is too short");
            }

            return new ValidationResult(x, sr);
        }

        private static float[][] Concatenate(float[][] a, float[][] b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("The number of rows in both arrays must be the same.");
            }

            int rows = a.Length;
            float[][] result = new float[rows][];

            for (int i = 0; i < rows; i++)
            {
                int lenA = a[i].Length;
                int lenB = b[i].Length;
                float[] row = new float[lenA + lenB];
                Buffer.BlockCopy(a[i], 0, row, 0, lenA * sizeof(float));
                Buffer.BlockCopy(b[i], 0, row, lenA * sizeof(float), lenB * sizeof(float));
                result[i] = row;
            }

            return result;
        }

        private static float[][] GetLastColumns(float[][] array, int contextSize)
        {
            int rows = array.Length;
            int cols = array[0].Length;

            if (contextSize > cols)
            {
                throw new ArgumentException("contextSize cannot be greater than the number of columns in the array.");
            }

            float[][] result = new float[rows][];

            for (int i = 0; i < rows; i++)
            {
                float[] row = new float[contextSize];
                Buffer.BlockCopy(array[i], (cols - contextSize) * sizeof(float), row, 0, contextSize * sizeof(float));
                result[i] = row;
            }

            return result;
        }

        public float[] Call(float[][] x, int sr)
        {
            var validation = ValidateInput(x, sr);
            x = validation.X;
            sr = validation.Sr;

            int numberSamples = (sr == 16000) ? 512 : 256;

            if (x[0].Length != numberSamples)
            {
                throw new ArgumentException($"Provided number of samples is {x[0].Length} (Supported values: 256 for 8000 sample rate, 512 for 16000)");
            }

            int batchSize = x.Length;
            int contextSize = (sr == 16000) ? 64 : 32;

            if (lastBatchSize == 0 || lastSr != sr || lastBatchSize != batchSize)
            {
                ResetStates();
            }

            if (context.Length != batchSize)
            {
                context = new float[batchSize][];

                for (int i = 0; i < batchSize; i++)
                {
                    context[i] = new float[contextSize];
                }
            }

            x = Concatenate(context, x);
            int rows = x.Length;
            int cols = x[0].Length;

            float[] inputData = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                Array.Copy(x[i], 0, inputData, i * cols, cols);
            }

            var inputTensor = new DenseTensor<float>(inputData, new[] { rows, cols });
            var srTensor = new DenseTensor<long>(new long[] { sr }, new[] { 1 });

            int stateDim0 = state.Length;
            int stateDim1 = state[0].Length;
            int stateDim2 = state[0][0].Length;

            float[] stateData = new float[stateDim0 * stateDim1 * stateDim2];
            int index = 0;

            for (int i = 0; i < stateDim0; i++)
            {
                for (int j = 0; j < stateDim1; j++)
                {
                    Array.Copy(state[i][j], 0, stateData, index, stateDim2);
                    index += stateDim2;
                }
            }

            var stateTensor = new DenseTensor<float>(stateData, new[] { stateDim0, stateDim1, stateDim2 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor),
                NamedOnnxValue.CreateFromTensor("sr", srTensor),
                NamedOnnxValue.CreateFromTensor("state", stateTensor)
            };

            using var outputs = session.Run(inputs);
            var outputTensor = outputs.First(o => o.Name == "output").AsTensor<float>();
            var newStateTensor = outputs.First(o => o.Name == "stateN").AsTensor<float>();

            context = GetLastColumns(x, contextSize);
            lastSr = sr;
            lastBatchSize = batchSize;

            var dims = newStateTensor.Dimensions;
            int d0 = dims[0], d1 = dims[1], d2 = dims[2];
            state = new float[d0][][];
            float[] newStateFlat = newStateTensor.ToArray();
            index = 0;

            for (int i = 0; i < d0; i++)
            {
                state[i] = new float[d1][];
                for (int j = 0; j < d1; j++)
                {
                    float[] row = new float[d2];
                    Array.Copy(newStateFlat, index, row, 0, d2);
                    state[i][j] = row;
                    index += d2;
                }
            }

            return outputTensor.ToArray();
        }
    }
}